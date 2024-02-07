import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        TP = np.diag(self.confusion_matrix)
        TN = self.confusion_matrix.sum() - (FP + FN + TP)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc
    def jc(self):
        jcc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum()-np.diag(self.confusion_matrix))
        return jcc
    def sen_e(self):
        FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        TP = np.diag(self.confusion_matrix)
        TN = self.confusion_matrix.sum() - (FP + FN + TP)

        teyixing = TN/(TN+FP)
        return  teyixing

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    def Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        # MIoU = np.nanmean(MIoU)
        return IoU

    def Precision(self):
        #  返回所有类别的精确率precision
        # precision = (self.confusion_matrix) / self.confusion_matrix.sum(axis = 0)
        # precision = np.nanmean(precision)

        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        TN = self.confusion_matrix.sum() - (FP + FN + TP)

        precision = TP / (TP + FP)
        return precision

    def Recall(self):
        #  返回所有类别的召回率recall
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1) ###axis=1 按行求和
        return recall
    
    def F1Score(self):
        precision = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 0)
        recall = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1)
        f1score = 2 * precision * recall / (precision + recall)
        return f1score

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




