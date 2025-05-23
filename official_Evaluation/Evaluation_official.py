### Evaluation
import os
import matplotlib.pyplot as plt
from torchcv.evaluations.coco import COCO
from torchcv.evaluations.eval_MR_multisetup import COCOeval

annType = 'bbox'
JSON_GT_FILE = os.path.join( '/home/urp6/workspace/ssd_for_KAISTPD/official_Evaluation/kaist_annotations_test20.json')
cocoGt = COCO(JSON_GT_FILE)

def evaluate_coco(test_json_path):

    fig_test,  ax_test  = plt.subplots(figsize=(18,15))
         
    rstFile = os.path.join(test_json_path)

    try:
        cocoDt = cocoGt.loadRes(rstFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.catIds  = [1]    
        cocoEval.evaluate(0)
        cocoEval.accumulate()
        curPerf = cocoEval.summarize(0)    
        cocoEval.draw_figure(ax_test, rstFile.replace('json', 'jpg'))        
        
        print('Recall: {:}'.format( 1-cocoEval.eval['yy'][0][-1] ) )

    except:
        import torchcv.utils.trace_error
        print('[Error] cannot evaluate by cocoEval. ')

if __name__ == '__main__':

    test_json_path = '/home/urp6/workspace/fusion_late/predict_late_ddssd/chal80.json'     #생성한 json파일 경로 넣기!
    evaluate_coco(test_json_path)
