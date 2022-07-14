from msilib.schema import Class
from typing import List, Optional, Tuple
from typing_extensions import Self
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.preprocessing import RobustScaler
import pybnesian as pybn
from itertools import product
import networkx as nx
from ufal.chu_liu_edmonds import chu_liu_edmonds
from IPython.display import SVG, display
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from sklearn.feature_selection import _mutual_info as skmi
import pickle
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
from hashlib import sha256
from pandas.util import hash_pandas_object
import pyarrow as pa
from sklearn.base import clone


def ext_hc(df,
           bn_type: type[pybn.BayesianNetworkType] = None,
           start: type[pybn.BayesianNetworkBase] = None,
           score: str = 'holdout-lik',
           bw_sel: Optional[str] = None,
           operators = ('arcs', 'node_type'),
           arc_blacklist: List[Tuple[str, str]] = [],
           arc_whitelist: List[Tuple[str, str]] = [],
           type_blacklist: List[Tuple[str, pybn.FactorType]] = [],
           type_whitelist: List[Tuple[str, pybn.FactorType]] = [],
           callback: type[pybn.Callback] = None,
           max_indegree: int = 0,
           max_iters: int = 2147483647,
           epsilon: float = 0,
           patience: int = 0,
           random_state: Optional[int] = None,
           num_folds: int = 10,
           test_holdout_ratio: float = 0.2,
           verbose: int = 0):
    if bn_type is not None and start is None:
        start = pybn.BayesianNetwork(type=bn_type, nodes=list(df.columns))
    elif bn_type is None and start is None:
        raise ValueError("No BN type or start BN")

    if bw_sel is not None:
        if bw_sel == 'scott':
            bw_sel = pybn.ScottsBandwidth()
        elif bw_sel == 'normal_reference':
            bw_sel = pybn.NormalReferenceRule()
        elif bw_sel == 'ucv':
            bw_sel = pybn.UCV()
        else:
            raise ValueError(
                'Only "scott", "normal_reference" or "ucv" allowed as bandwidth selectors')
    else:
        bw_sel = pybn.UCV()

    arguments = pybn.Arguments(dict_arguments={
        pybn.CKDEType(): {'bandwidth_selector': bw_sel}})

    if score == 'holdout-lik':
        score = pybn.HoldoutLikelihood(
            df, test_ratio=test_holdout_ratio, seed=random_state, construction_args=arguments)
    elif score == 'cv-lik':
        score = pybn.CVLikelihood(
            df=df, k=num_folds, seed=random_state, construction_args=arguments)
    elif score == 'validated-lik':
        score = pybn.ValidatedLikelihood(
            df, test_ratio=test_holdout_ratio, k=num_folds, seed=random_state, construction_args=arguments)
    elif score == 'bic':
        score = pybn.BIC(df)
    elif score == 'bge':
        score = pybn.BGe(df)
    elif score == 'bde':
        score = pybn.BDe(df)
    else:
        raise ValueError(
            'Only "holdout-lik", "cv-lik", "validated-lik", "bic", "bge" or "bde" are allowed scores')

    ops = []
    if 'arcs' in operators:
        ops.append(pybn.ArcOperatorSet(
            blacklist=arc_blacklist, whitelist=arc_whitelist, max_indegree=max_indegree))
    if 'node_type' in operators:
        ops.append(pybn.ChangeNodeTypeSet(
            type_blacklist=type_blacklist, type_whitelist=type_whitelist))
    operators = pybn.OperatorPool(opsets=ops)

    bn = pybn.GreedyHillClimbing().estimate(
        operators=operators, score=score, start=start, arc_blacklist=arc_blacklist, arc_whitelist=arc_whitelist,
        type_blacklist=type_blacklist, type_whitelist=type_whitelist, callback=callback, max_indegree=max_indegree,
        max_iters=max_iters, epsilon=epsilon, patience=patience, verbose=verbose
    )
    return bn


# https://gist.github.com/dsevero/3f3db7acb45d6cd8e945e8a32eaca168
class HashableDataFrame(pd.DataFrame):
    def __init__(self, obj):
        super().__init__(obj)

    def __hash__(self):
        hash_value = sha256(hash_pandas_object(
            self, index=True).values).hexdigest()
        return hash(hash_value)

    def __eq__(self, other):
        return self.equals(other)


_prep_X_predict_proba_cache = {}


def _prep_X_predict_proba(classnames, classlevels, feature_names_in, X):
    # Check if dataframe preparation for predict was already computed,
    # avoiding multiple pyarrow converisons
    X = HashableDataFrame(X)
    # cache_entry = X.__hash__()
    # ret = _prep_X_predict_proba_cache.get(cache_entry, None)
    # if ret is not None:
    #     # print('Cache used!')
    #     return ret

    X['_sample'] = range(len(X))
    value_combinations = list(product(*classlevels))
    X['_target'] = [value_combinations] * len(X)
    X = X.explode('_target')
    X[classnames] = pd.DataFrame(
        X['_target'].tolist(), index=X.index)
    X = X.astype({
        classnames[i]: pd.api.types.CategoricalDtype(
            categories=classlevels[i])
        for i in range(len(classnames))
    })
    X_pa = pa.RecordBatch.from_pandas(X[feature_names_in + classnames])
    X_meta = X[['_sample'] + classnames]
    # _prep_X_predict_proba_cache[cache_entry] = (X_meta, X_pa)
    return X_meta, X_pa


class BNClassifier(BaseEstimator, ClassifierMixin, MultiOutputMixin):
    """
    Base class for Bayesian Network classifier
    """

    def __init__(self,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)
        MultiOutputMixin.__init__(self)
        self.greedy_arc_remove = greedy_arc_remove
        self.greedy_node_remove = greedy_node_remove
        self.greedy_prune_cv = greedy_prune_cv
        self.greedy_prune_impatient = greedy_prune_impatient
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.bn_ = None
        

    def predict_proba_dep(self, X: type[pd.DataFrame]):
        """
        Predict taking into account dependendencies between classes, output is a pandas series
        of probabilities with hierarchical index: (_sample, class1, ..., classN)
        """
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, columns=self.feature_names_in_).astype(self.dtypes_, copy=False)
        if self.bn_ is None:
            raise ValueError("Model not fitted")
        X_meta, X_pa = _prep_X_predict_proba(
            self.classnames_, self.classes_, self.feature_names_in_, X)
        X_meta['_lik'] = np.e ** self.bn_.logl(X_pa)
        X_meta['_lik'] = X_meta['_lik'] / \
            X_meta.groupby('_sample')['_lik'].transform('sum')
        return pd.Series(X_meta['_lik'].values, index=pd.MultiIndex.from_frame(
            X_meta[['_sample'] + self.classnames_].reindex()
        ))

    def predict(self, X: type[pd.DataFrame]):
        """
        Predict taking into account dependendencies between classes
        """
        pred = self.predict_proba_dep(X)
        pred = pred[pred.groupby(level='_sample').idxmax()].index.to_frame(
            index=False).drop('_sample', axis=1)
        return pred

    def predict_proba(self, X: type[pd.DataFrame]):
        """
        Predict probabilities without taking into account dependendencies between classes
        """
        probs = self.predict_proba_dep(X)
        out = []
        for cl in self.classnames_:
            levels = ['_sample'] + [cl]
            out.append(
                probs.groupby(level=levels).sum().unstack(level=cl).values
            )
        if len(out) == 1:
            return out[0]
        else:
            return out

    def decision_function(self, X: type[pd.DataFrame]):
        return self.predict_proba(X)[:, 1]

    def predict_log_proba(self, X: type[pd.DataFrame]):
        """
        Predict log probabilities without taking into account dependendencies between classes
        """
        return [np.log(x) for x in self.predict_proba(X)]

    def as_pydot(self):
        DG = nx.DiGraph()
        DG.add_nodes_from(self.bn_.nodes())
        DG.add_edges_from(self.bn_.arcs())
        for node in DG.nodes:
            if self.bn_.node_type(node) == pybn.CKDEType():
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'
            elif self.bn_.node_type(node) == pybn.DiscreteFactorType():
                DG.nodes[node]['style'] = 'dashed'
                DG.nodes[node]['fillcolor'] = 'gray'
        return nx.nx_pydot.to_pydot(DG)

    def auc(self, X, y_true):
        y_scores = self.predict_proba(X)
        if len(y_scores.shape) == 2:
            y_scores = [y_scores]

        def bin(x):
            if x.shape[1] == 2:
                return x[:, 1]
            return x

        return {
            c: roc_auc_score(y_true[c], bin(y_scores[i]), multi_class='ovr')
            for i, c in enumerate(self.classnames_)
        }

    def display(self):
        pdot = self.as_pydot()
        plt = SVG(pdot.create_svg())
        display(plt)

    def __str__(self) -> str:
        return 'Bayesian Network Classifier'

    def __repr__(self, *args, **kwargs):
        return str(self)

    def _greedy_arc_remove(self, X, y):
        kf = StratifiedKFold(n_splits=self.greedy_prune_cv,
                             shuffle=True, random_state=self.random_state)
        curr_best_score = self._cross_val_score(self.bn_, X, y, kf)
        it = 1
        i = 0
        while True:
            improve = False
            arcs = self.bn_.arcs()
            node_types = list(self.bn_.node_types().items())
            if self.verbose >= 1:
                print(
                    f'Iteration {it}, {len(arcs)} arcs, {curr_best_score} current best score')
            remove_order = list(range(i, len(arcs))) + list(range(0, i))
            for i in remove_order:
                candidate_arcs = [arcs[j] for j in range(len(arcs)) if j != i]
                candidate_nodes = set(a[0] for a in candidate_arcs).union(
                    a[1] for a in candidate_arcs)
                candidate_node_types = [
                    t for t in node_types if t[0] in candidate_nodes]
                candidate_bn = pybn.BayesianNetwork(type=self.bn_type_,
                    nodes=list(candidate_nodes), arcs=candidate_arcs, node_types=candidate_node_types)
                score = self._cross_val_score(candidate_bn, X, y, kf)
                if self.verbose >= 2:
                    print(f'{i+1}/{len(arcs)}: {score} <? {curr_best_score}')
                if score > curr_best_score:
                    curr_best_score = score
                    self.bn_ = candidate_bn
                    improve = True
                if improve and self.greedy_prune_impatient:
                    break
            if not improve:
                break
            it += 1
        return self

    def _greedy_node_remove(self, X, y):
        kf = StratifiedKFold(n_splits=self.greedy_prune_cv,
                             shuffle=True, random_state=self.random_state)
        curr_best_score = self._cross_val_score(self.bn_, X, y, kf)
        it = 1
        i = 0
        while True:
            improve = False
            arcs = self.bn_.arcs()
            node_types = list(self.bn_.node_types().items())
            if self.verbose >= 1:
                print(
                    f'Iteration {it}, {len(node_types)} nodes, {curr_best_score} current best score')
            remove_order = list(range(i, len(node_types))) + list(range(0, i))
            for i in remove_order:
                removed_node = node_types[i][0]
                if (self.bn_.num_parents(removed_node) + self.bn_.num_children(removed_node)) == 0:
                    continue
                candidate_node_types = [node_types[j]
                                        for j in range(len(node_types)) if j != i]
                candidate_nodes = set(cn[0] for cn in candidate_node_types)
                candidate_arcs = [arcs[j] for j in range(len(arcs))
                                  if (arcs[j][0] in candidate_nodes and arcs[j][1] in candidate_nodes)]
                candidate_bn = pybn.BayesianNetwork(type=self.bn_type_,
                    nodes=list(candidate_nodes), arcs=candidate_arcs, node_types=candidate_node_types)
                score = self._cross_val_score(candidate_bn, X, y, kf)
                if self.verbose >= 2:
                    print(f'{i+1}/{len(node_types)}: {score} <? {curr_best_score}')
                if score > curr_best_score:
                    curr_best_score = score
                    self.bn_ = candidate_bn
                    improve = True
                if improve and self.greedy_prune_impatient:
                    break
            if not improve:
                break
            it += 1
        return self

    def _score_f(self, bn, tr_index, vl_index, X, y):
        self.bn_ = pybn.SemiparametricBN(
            graph=bn.graph(),
            node_types=list(bn.node_types().items()))
        X_train, X_val = X.iloc[tr_index], X.iloc[vl_index]
        y_train, y_val = y.iloc[tr_index], y.iloc[vl_index]
        self.bn_.fit(pd.concat([X_train, y_train], axis=1))
        return self.auc(X_val, y_val)[y.columns[0]]

    def _cross_val_score(self, bn, X, y, cv):
        scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(__class__._score_f)(
                clone(self), bn, tr_index, vl_index, X, y)
            for tr_index, vl_index in cv.split(X, y))
        return np.mean(scores)

    def _apply_greedy_prune(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        if self.greedy_node_remove:
            self._greedy_node_remove(X, y)
        if self.greedy_arc_remove:
            self._greedy_arc_remove(X, y)
        return self

    def get_data_info(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        self.classnames_ = list(y.columns)
        self.feature_names_in_ = list(X.columns)
        self.classes_ = [tuple(y[i].dtype.categories)
                             for i in self.classnames_]
        self.dtypes_ = {col: X[col].dtype for col in self.feature_names_in_}

    def get_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        raise NotImplementedError

    def refine_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        raise NotImplementedError

    def fit_final_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        df = pd.concat([X, y], axis=1)
        self._apply_greedy_prune(X, y)
        self.bn_.fit(df)
        return self

    def fit(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        if type(y) == pd.Series:
            y = pd.DataFrame({y.name: y})
        self.get_data_info(X, y)
        self.get_initial_structure(X, y)
        self.refine_initial_structure(X, y)
        self.fit_final_structure(X, y)
        return self


class HCBNClassifier(BNClassifier):
    VALID_SCORES = ['bde', 'bic', 'bge', 'holdout-lik', 'validated-lik', 'cv-lik']

    def __init__(self,
                 bn_score: str = 'holdout-lik',
                 bw_sel: Optional[str] = None,
                 operators = ('arcs', 'node_type'),
                 max_indegree: int = 0,
                 max_iters: int = 2147483647,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        super().__init__(
            greedy_node_remove=greedy_node_remove, greedy_arc_remove=greedy_arc_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        assert bn_score in self.VALID_SCORES, f'Score should be one of {", ".join(self.VALID_SCORES)}'
        self.initial_struc_ = None
        self.bn_type_ = None
        self.bn_score = bn_score
        self.bw_sel = bw_sel
        self.operators = operators
        self._arc_blacklist = []
        self._arc_whitelist = []
        self._type_whitelist = []
        self._type_blacklist = []
        self._callback = None
        self.max_indegree = max_indegree
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.patience = patience
        self.test_holdout_ratio = test_holdout_ratio
        self.num_folds = num_folds


    def get_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        self.start_ = pybn.BayesianNetwork(type=self.bn_type_, nodes=(self.classnames_ + self.feature_names_in_))
        
        
    def bn_concrete(self, bn, df):
        bn.set_unknown_node_types(df, self._type_blacklist)
        node_types = list(bn.node_types().items())
        if bn.type() == pybn.CLGNetworkType():
            bn = pybn.CLGNetwork(graph=bn.graph(), node_types=node_types)
        elif bn.type() == pybn.SemiparametricBNType():
            bn = pybn.SemiparametricBN(graph=bn.graph(), node_types=node_types)
        elif bn.type() == pybn.DiscreteBNType():
            bn = pybn.DiscreteBN(graph=bn.graph(), node_types=node_types)
        return bn


    def refine_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        if not self.operators or not self.max_iters:
            self.bn_ = pybn.BayesianNetwork(type=self.start_.type(), graph=self.start_.graph())
            return self
        df = pd.concat([X, y], axis=1)
        self.bn_ = ext_hc(
            df=df,
            start=self.bn_concrete(self.start_, df),
            score=self.bn_score,
            bw_sel=self.bw_sel,
            operators=self.operators,
            arc_blacklist=self._arc_blacklist,
            arc_whitelist=self._arc_whitelist,
            type_whitelist=self._type_whitelist,
            type_blacklist=self._type_blacklist,
            callback=self._callback,
            max_indegree=self.max_indegree,
            max_iters=self.max_iters,
            epsilon=self.epsilon,
            patience=self.patience,
            random_state=self.random_state,
            test_holdout_ratio=self.test_holdout_ratio,
            num_folds=self.num_folds,
            verbose=self.verbose,
        )
        return self


class HCClassifierCLG(HCBNClassifier):
    VALID_SCORES = ['bic', 'holdout-lik', 'validated-lik', 'cv-lik']

    def __init__(self,
                 bn_score: str = 'bic',
                 max_indegree: int = 0,
                 max_iters: int = 2147483647,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        assert bn_score in self.VALID_SCORES, f'Score should be one of {", ".join(self.VALID_SCORES)}'
        super().__init__(
            bn_score=bn_score, operators=('arcs', ),
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            random_state=random_state, num_folds=num_folds, test_holdout_ratio=test_holdout_ratio,
            verbose=verbose, greedy_arc_remove=greedy_arc_remove, greedy_node_remove=greedy_node_remove,
            greedy_prune_cv=greedy_prune_cv, greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs)
        self.bn_type_ = pybn.CLGNetworkType()


class HCClassifierSP(HCBNClassifier):
    VALID_SCORES = ['holdout-lik', 'validated-lik', 'cv-lik']

    def __init__(self,
                 bn_score: str = 'holdout-lik',
                 bw_sel: str = 'ucv',
                 operators = ('arcs', 'node_type'),
                 max_indegree: int = 0,
                 max_iters: int = 2147483647,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        assert bn_score in self.VALID_SCORES, f'Score should be one of {", ".join(self.VALID_SCORES)}'
        super().__init__(
            bw_sel=bw_sel, bn_score=bn_score,
            operators=operators, max_indegree=max_indegree,
            max_iters=max_iters, epsilon=epsilon, patience=patience, random_state=random_state,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, verbose=verbose,
            greedy_arc_remove=greedy_arc_remove, greedy_node_remove=greedy_node_remove,
            greedy_prune_cv=greedy_prune_cv, greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs)
        self.bn_type_ = pybn.SemiparametricBNType()


class HCClassifierDisc(HCBNClassifier):
    VALID_SCORES = ['bic', 'bde', 'holdout-lik', 'validated-lik', 'cv-lik']

    def __init__(self,
                 bn_score: str = 'bde',
                 max_indegree: int = 0,
                 max_iters: int = 2147483647,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        assert bn_score in self.VALID_SCORES, f'Score should be one of {", ".join(self.VALID_SCORES)}'
        super().__init__(
            bn_score=bn_score, operators=('arcs', ),
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            random_state=random_state, num_folds=num_folds, test_holdout_ratio=test_holdout_ratio,
            verbose=verbose, greedy_arc_remove=greedy_arc_remove, greedy_node_remove=greedy_node_remove,
            greedy_prune_cv=greedy_prune_cv, greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs)
        self.bn_type_ = pybn.DiscreteBNType()

_cmi_cache = {}
class KNNCMIStats:
    """
    Auxiliar class for conditional mutual independence for BN Classifiers
    """

    @staticmethod
    def cmi(df, x, y, z=None, k=3):
        if z is None:
            X = df[x].values
            Y = df[y].values
            k_p = min(k, len(X) // 2)
            x_discrete=X.dtype == 'category'
            y_discrete=Y.dtype == 'category'
            return skmi._compute_mi(x=X, y=Y, x_discrete=x_discrete, y_discrete=y_discrete, n_neighbors=k_p)
        Z = df[z]
        mi = 0
        for z_val in Z.dtype.categories:
            partition = Z == z_val
            p_z_val = (partition).sum() / len(Z)
            X = df.loc[partition, x].values
            Y = df.loc[partition, y].values
            k_p = min(k, len(X) // 2)
            mi += skmi._compute_mi(
                x=X, y=Y, x_discrete=X.dtype == 'category', y_discrete=Y.dtype == 'category', n_neighbors=k_p
            ) * p_z_val
        return mi

    def compute(self, df, predictors, class_name, k=3, verbose=0, n_jobs=-1):
        df = HashableDataFrame(df)
        cache_entry = (df, tuple(predictors), tuple(class_name), k).__hash__()
        ret = _cmi_cache.get(cache_entry, None)
        if ret is not None:
            self.mi_wr_class, self.mi_gv_class = ret
            return self
        if verbose:
            print('Computing MI(predictors, class)...')
        # Estandarizar con la mediana
        numeric = df[predictors].columns[df[predictors].dtypes != 'category']
        if len(numeric) > 0:
            df.loc[:, numeric] = df.loc[:, numeric].apply(
                lambda col: RobustScaler().fit_transform(col.values.reshape(-1, 1))[:,0])

        self.mi_wr_class = np.empty_like(predictors)
        self.mi_wr_class[:] = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.cmi)(df[[p, class_name]], p, class_name, None, k) for p in predictors)
        self.mi_wr_class = self.mi_wr_class.astype(float)
        if verbose:
            print('Computing CMI(predictors|class)')
        self.mi_gv_class = np.empty((len(predictors), len(predictors)))
        pairs = [(predictors[i], predictors[j]) for i in range(
            len(predictors)) for j in range(i+1, len(predictors))]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.cmi)(df[[p1, p2, class_name]], p1, p2, class_name, k) for p1, p2 in pairs)
        idx = 0
        for i in range(len(predictors)):
            self.mi_gv_class[i, i] = 0
            for j in range(i+1, len(predictors)):
                self.mi_gv_class[i, j] = results[idx]
                self.mi_gv_class[j, i] = results[idx]
                idx += 1
        _cmi_cache[cache_entry] = np.copy(
            self.mi_wr_class), np.copy(self.mi_gv_class)
        return self


class KDBBNClassifierMixin():
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 k=1):
        self.mi_stats_ = None
        self.mi_nneighbors = mi_nneighbors
        self.mi_thres = mi_thres
        self.k = k

    def get_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        df = pd.concat([X, y], axis=1)
        predictors = self.feature_names_in_
        class_name = self.classnames_[0]
        if self.mi_stats_ is None:
            self.mi_stats_ = KNNCMIStats().compute(
                df=df, predictors=predictors, class_name=class_name, k=self.mi_nneighbors,
                n_jobs=self.n_jobs, verbose=self.verbose)
        dag = pybn.Dag(nodes=predictors+[class_name])
        in_model = set()
        inclusion_order = np.argsort(-1 * self.mi_stats_.mi_wr_class)
        categorical = X.columns[X.dtypes == 'category']
        for i in inclusion_order:
            p = predictors[i]
            dag.add_arc(class_name, p)
            in_model.add(p)
            for _ in range(min(self.k, len(in_model)-1)):
                idx_mx = -1
                for l in range(len(predictors)):
                    l_p = predictors[l]
                    if l_p == p or (p in categorical and not l_p in categorical) or not l_p in in_model or dag.has_arc(l_p, p):
                        continue
                    if idx_mx == -1 or self.mi_stats_.mi_gv_class[i, l] > self.mi_stats_.mi_gv_class[i, idx_mx]:
                        idx_mx = l
                if idx_mx == -1:
                    break
                dag.add_arc(predictors[idx_mx], p)
        for i in range(len(predictors)):
            if self.mi_stats_.mi_wr_class[i] < self.mi_thres:
                for p in dag.parents(predictors[i]):
                    dag.remove_arc(p, predictors[i])
                for c in dag.children(predictors[i]):
                    dag.remove_arc(predictors[i], c)
        self.start_ = pybn.BayesianNetwork(type=self.bn_type_, graph=dag)


class KDBBNClassifierCLG(KDBBNClassifierMixin, HCClassifierCLG):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 k=1,
                 bn_score: str = 'bic',
                 max_indegree: int = 0,
                 max_iters: int = 0,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierCLG.__init__(self,
            bn_score=bn_score,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose
        )
        KDBBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres, k=k)


class KDBBNClassifierSP(KDBBNClassifierMixin, HCClassifierSP):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 k=1,
                 bn_score: str = 'cv-lik',
                 bw_sel: str = 'ucv',
                 max_indegree: int = 0,
                 operators = ('node_type', ),
                 max_iters: int = 0,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierSP.__init__(self,
            bn_score=bn_score, bw_sel=bw_sel,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs, operators=operators,
            random_state=random_state, verbose=verbose
        )
        KDBBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres, k=k)


class KDBBNClassifierDisc(KDBBNClassifierMixin, HCClassifierDisc):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 k=1,
                 bn_score: str = 'bde',
                 max_indegree: int = 0,
                 operators = (),
                 max_iters: int = 0,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierDisc.__init__(self,
            bn_score=bn_score,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs, operators=operators,
            random_state=random_state, verbose=verbose
        )
        KDBBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres, k=k)


class CLTANBNClassifierMixin():
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0):
        self.mi_stats_ = None
        self.mi_nneighbors = mi_nneighbors
        self.mi_thres = mi_thres

    @staticmethod
    def heads_to_dag(predictors, heads):
        dag = pybn.Dag()
        for i_child, i_parent in enumerate(heads):
            child = predictors[i_child]
            if not dag.contains_node(child):
                dag.add_node(child)
            if i_parent != -1:
                parent = predictors[i_parent]
                if not dag.contains_node(parent):
                    dag.add_node(parent)
                dag.add_arc(parent, child)
        return dag

    def get_initial_structure(self, X: type[pd.DataFrame], y: type[pd.DataFrame]):
        df = pd.concat([X, y], axis=1)
        predictors = self.feature_names_in_
        class_name = self.classnames_[0]
        if self.mi_stats_ is None:
            self.mi_stats_ = KNNCMIStats().compute(
                df=df, predictors=predictors, class_name=class_name, k=self.mi_nneighbors,
                n_jobs=self.n_jobs, verbose=self.verbose)
        categorical = X.columns[X.dtypes == 'category']
        score_matrix = np.copy(self.mi_stats_.mi_gv_class)
        for i in range(len(predictors)):
            for j in range(len(predictors)):
                if i == j:
                    continue
                if predictors[j] not in categorical and predictors[i] in categorical:
                    # j no puede ser un padre continuo del hijo i categórico
                    # Convención (i,j) es j -> i
                    score_matrix[i, j] = np.nan
        heads, _ = chu_liu_edmonds(score_matrix)
        dag = self.heads_to_dag(predictors, heads)
        dag.add_node(class_name)
        for p in predictors:
            dag.add_arc(class_name, p)
        self.start_ = pybn.BayesianNetwork(type=self.bn_type_, graph=dag)


class CLTANBNClassifierCLG(CLTANBNClassifierMixin, HCClassifierCLG):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 bn_score: str = 'bic',
                 max_indegree: int = 0,
                 max_iters: int = 0,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierCLG.__init__(self,
            bn_score=bn_score,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose
        )
        CLTANBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres)


class CLTANBNClassifierSP(CLTANBNClassifierMixin, HCClassifierSP):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 bn_score: str = 'cv-lik',
                 bw_sel: str = 'ucv',
                 max_indegree: int = 0,
                 operators = ('node_type'),
                 max_iters: int = 2147483647,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierSP.__init__(self,
            bn_score=bn_score, bw_sel=bw_sel,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs, operators=operators,
            random_state=random_state, verbose=verbose
        )
        CLTANBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres)


class CLTANBNClassifierDisc(CLTANBNClassifierMixin, HCClassifierDisc):
    def __init__(self,
                 mi_nneighbors: Optional[int] = 10,
                 mi_thres: float = 0.0,
                 bn_score: str = 'bde',
                 max_indegree: int = 0,
                 operators = (),
                 max_iters: int = 0,
                 epsilon: float = 0,
                 patience: int = 0,
                 num_folds: int = 10,
                 test_holdout_ratio: float = 0.2,
                 greedy_arc_remove: bool = False,
                 greedy_node_remove: bool = False,
                 greedy_prune_cv: int = 5,
                 greedy_prune_impatient: bool = True,
                 random_state: Optional[int] = None,
                 n_jobs: Optional[float] = -1,
                 verbose: int = 0):
        HCClassifierDisc.__init__(self,
            bn_score=bn_score,
            max_indegree=max_indegree, max_iters=max_iters, epsilon=epsilon, patience=patience,
            num_folds=num_folds, test_holdout_ratio=test_holdout_ratio, greedy_arc_remove=greedy_arc_remove,
            greedy_node_remove=greedy_node_remove, greedy_prune_cv=greedy_prune_cv,
            greedy_prune_impatient=greedy_prune_impatient, n_jobs=n_jobs, operators=operators,
            random_state=random_state, verbose=verbose
        )
        CLTANBNClassifierMixin.__init__(self, mi_nneighbors=mi_nneighbors, mi_thres=mi_thres)