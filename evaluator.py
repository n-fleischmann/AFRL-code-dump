import torch
import torch.nn as nn

import pytorch_ood


from data import course_ood, dogs_places_cars, testing_id, testing_ood


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class Evaluator:


    def __init__(self, model: nn.Module, detector: pytorch_ood.detector, dataset: str, split: str, batch_size: int, im_size: str) -> None:
        self.model = model
        self.model.eval()
        self.model.to(DEVICE)

        self.detector = detector
    
        self.model_detector = self.detector(self.model)

        self.im_size = im_size
        self.batch_size = batch_size
        self.split = split
        self.dataset = dataset

        self.course_ood_dataloaders = dogs_places_cars(self.im_size, self.batch_size)
        self.course_ood_names = ['dogs', 'places', 'cars', 'all']

        self.id_planes = testing_id(self.dataset, self.split, self.batch_size, self.im_size)

        if self.dataset == 'planes':
            self.levels = [1, 2, 3, -1]
        elif self.dataset == 'ships':
            self.levels = [1, 2, -1]

    
    def id_accuracy(self, limit=-1):

        running_correct = 0.0
        running_total = 0.0

        with torch.no_grad():
            for i, (X, y) in enumerate(self.id_planes):
                if limit != -1 and i >= limit:
                    break

                X, y = X.to(DEVICE), y.to(DEVICE)

                _, preds = self.model(X).softmax(dim=1).max(dim=1)

                running_correct += preds.eq(y).sum().item()
                running_total += y.shape[0]

        return running_correct / running_total


    @staticmethod
    def _eval(model_detector, id_loader, ood_loaders):

        if len(ood_loaders) == 1:
            metrics = [pytorch_ood.utils.OODMetrics()]
        else:
            metrics = [pytorch_ood.utils.OODMetrics() for _ in range(len(ood_loaders) + 1)]


        with torch.no_grad():
            for X, y in id_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                output = model_detector(X)
                for metric in metrics:
                    metric.update(output, y)

            for i, loader in enumerate(ood_loaders):
                for X, y in loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    output = model_detector(X)
                    metrics[i].update(output, y)
                    metrics[-1].update(output, y)

        return [metric.compute() for metric in metrics]

    
    def holdout_vs_ID_eval(self):

        ood_loaders = [testing_ood(self.dataset, self.split, self.batch_size, self.im_size, level) for level in self.levels[:-1]]
        
        metrics = self._eval(self.model_detector, self.id_planes, ood_loaders)
        all_results = {level: metric for level, metric in zip(self.levels, metrics)}

        return all_results


    def course_vs_fg_eval(self):
        ood_planes = testing_ood(self.dataset, self.split, self.batch_size, self.im_size, -1, set_unknown=False)

        metrics = self._eval(self.model_detector, ood_planes, self.course_ood_dataloaders)
        all_results = {name: metric for name, metric in zip(self.course_ood_names, metrics)}

        return all_results


    def __call__(self):
        
        results = [self.id_accuracy()]

        holdout_results = self.holdout_vs_ID_eval()

        results += [holdout_results[key]['AUROC'] for key in holdout_results.keys()]
        results += [holdout_results[key]['FPR95TPR'] for key in holdout_results.keys()]

        course_fg_results = self.course_vs_fg_eval()

        results += [course_fg_results[key]['AUROC'] for key in course_fg_results.keys()]
        results += [course_fg_results[key]['FPR95TPR'] for key in course_fg_results.keys()]


        return results
    


    









    
