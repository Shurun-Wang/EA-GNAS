from core.utiles import *
from sklearn.utils import resample

def SCH_HC(site='all', Balance_flag=True):
    KTT_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KTT/HC.npy', allow_pickle=True)
    KTT_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KTT/SCH.npy', allow_pickle=True)
    KUT_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KUT/HC.npy', allow_pickle=True)
    KUT_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/KUT/SCH.npy', allow_pickle=True)
    SWA_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/SWA/HC.npy', allow_pickle=True)
    SWA_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/SWA/SCH.npy', allow_pickle=True)
    UTO_HC = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/UTO/HC.npy', allow_pickle=True)
    UTO_SCH = np.load('/Users/wangsr/PycharmProjects/graph_fc/data/HC_SCH/UTO/SCH.npy', allow_pickle=True)
    # hc = np.load('data/HC_MDD/hc.npy', allow_pickle=True)
    # mdd = np.load('data/HC_MDD/mdd.npy', allow_pickle=True)
    if Balance_flag:
        KTT_HC = resample(KTT_HC, replace=False, n_samples=KTT_SCH.shape[0], random_state=67)
        KUT_HC = resample(KUT_HC, replace=False, n_samples=KUT_SCH.shape[0], random_state=67)
        SWA_HC = resample(SWA_HC, replace=False, n_samples=SWA_SCH.shape[0], random_state=67)
        UTO_HC = resample(UTO_HC, replace=False, n_samples=UTO_SCH.shape[0], random_state=67)
        # hc = resample(hc, replace=False, n_samples=mdd.shape[0], random_state=67)
    # SWA_HC = np.load('data/HC_MDD/hc.npy', allow_pickle=True)
    # SWA_MDD = np.load('data/HC_MDD/mdd.npy', allow_pickle=True)
    if site == 'all':
        HC = np.concatenate([KTT_HC, KUT_HC, SWA_HC, UTO_HC])
        SCH = np.concatenate([KTT_SCH, KUT_SCH, SWA_SCH, UTO_SCH])
    elif site == 'KTT':
        HC, SCH = KTT_HC, KTT_SCH
    elif site == 'KUT':
        HC, SCH = KUT_HC, KUT_SCH
    elif site == 'SWA':
        HC, SCH = SWA_HC, SWA_SCH
    elif site == 'UTO':
        HC, SCH = UTO_HC, UTO_SCH
    else:
        raise NotImplementedError
    return HC, SCH


seed_everything(6767)
HC, PA = SCH_HC(site='all', Balance_flag=True)

HC_label, PA_label = np.zeros(HC.shape[0], dtype=int), np.ones(PA.shape[0], dtype=int)
X = np.concatenate([HC, PA], axis=0)
y = np.concatenate([HC_label, PA_label], axis=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# pca.fit(x_train)
# # 对数据进行降维
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)
#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# clf = LogisticRegression()
# clf = SVC(kernel='linear')
# clf = RandomForestClassifier()
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall_score = recall_score(y_test, y_pred)
print('f1', f1)
print('precision', precision)
print('auc', auc)
print('accuracy', accuracy)
print('recall_score', recall_score)