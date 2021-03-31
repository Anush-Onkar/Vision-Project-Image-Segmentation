#Project Submission by Anush Onkarappa (7010620- anon00001@stud.uni-saarland.de) & Hitesh Kotte (7010571-hiko00001@stud.uni-saarland.de)
from sklearn import metrics

def evaluate(ground_truth, predictions):
    ground_truth = torch.flatten(ground_truth, start_dim = 0, end_dim = 2)
    pred = torch.argmax(predictions, dim=1)
    pred = torch.flatten(pred, start_dim = 0, end_dim = 2)
    ground_truth = ground_truth.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    f1_score = metrics.f1_score(ground_truth, pred,average = 'micro')
    sensitivity=metrics.recall_score(ground_truth,pred,average = 'weighted')
    jaccard_score=metrics.jaccard_score(ground_truth,pred,average = 'weighted')
    accuracy=metrics.accuracy_score(ground_truth,pred)
    return f1_score,sensitivity,jaccard_score,accuracy