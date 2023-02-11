from .builder import DATASETS 
from .custom import CustomDataset 
 
#将 Manipulation 类注册到 DATASETS 里 
@DATASETS.register_module() 
class ManipulationDataset(CustomDataset): 
    # 数据集标注的各类名称，即 0, 1, 2, 3... 各个类别的对应名称 
    CLASSES = ('background', 'Manipulation') 
    # 各类类别的 BGR 三通道值，用于可视化预测结果 
    PALETTE = [[0, 0, 0], [255, 255, 255]] 
 
    # 图片和对应的标注，这里对应的文件夹下均为 .png 后缀 
    def __init__(self, **kwargs): 
        super(ManipulationDataset, self).__init__( 
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False, # False : 此时 label 里的 0（对应上面 CLASSES 里第一个 类名）在计算损失函数和指标时不会被忽略。 
            **kwargs) 