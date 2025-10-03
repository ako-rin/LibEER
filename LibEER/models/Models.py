from .DGCNN import DGCNN
from .RGNN_official import SymSimGCNNet
from .EEGNet import EEGNet
from .STRNN import STRNN
from .GCBNet import GCBNet
from .DBN import DBN
from .TSception import TSception
from .SVM import SVM
from .CDCN import CDCN
from .HSLT import HSLT
from .ACRNN import ACRNN
from .GCBNet_BLS import GCBNet_BLS
from .MsMda import MSMDA

Model = {
    'DGCNN': DGCNN,
    'RGNN_official': SymSimGCNNet,
    'GCBNet': GCBNet,
    'GCBNet_BLS': GCBNet_BLS,
    'CDCN': CDCN,
    'DBN': DBN,
    'STRNN': STRNN,
    'EEGNet': EEGNet,
    'HSLT': HSLT,
    'ACRNN': ACRNN,
    'TSception': TSception,
    'MsMda': MSMDA, 'MSMDA': MSMDA,
    'svm': SVM,
}
