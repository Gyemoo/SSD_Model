from utils import *
from datasets import KAISTPDDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
#keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 16
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '/home/urp6/workspace/fusion_late/checkpoint_fusion_ssd_kor/checkpoint_KAISTPD79.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = KAISTPDDataset(data_folder,
                                split='test',)
                                #keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    #true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels,) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)


            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.2, max_overlap=0.5,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
            

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            #difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            #true_boxes.extend(boxes)
            #true_labels.extend(labels)
            #true_difficulties.extend(difficulties)
        

        
        # Calculate mAP
        #APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,)# true_difficulties)
    

    

    

    submit = []

    for i in range(len(det_boxes)): # 1 image

        for j in range(len(det_boxes[i])): # 1 det box
            tmp = dict()
            tmp['image_id'] = i
            tmp['category_id'] = det_labels[i][j].detach().cpu().numpy().item()
            val = det_boxes[i][j].detach().cpu().numpy().tolist() # use xmin,ymin then
            
            
            val[0] = val[0]*640
            val[1] = val[1]*512
            val[2] = val[2]*640
            val[3] = val[3]*512 # xmin ymin xmax ymax

            val[2] = val[2]-val[0]
            val[3] = val[3]-val[1]


                

            tmp['bbox'] = val
            tmp['score'] = det_scores[i][j].detach().cpu().numpy().item()
            submit.append(tmp)
    with open('/home/urp6/workspace/fusion_late/predict_late_ddssd/chal80.json','w') as j:
        json.dump(submit,j)


    #import pdb; pdb.set_trace() # all detections
    print()
    # Print AP for each class
    #pp.pprint(APs)

    #print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
