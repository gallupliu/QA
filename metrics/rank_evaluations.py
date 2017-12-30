import numpy as np
import math

class rank_eval():

    def __init__(self, qids,labels,scores,metrics,rel_threshold=0.):
        self.rel_threshold = rel_threshold
        self.scores_dict = {}
        self.labels_dict = {}


        length = len(qids)
        for i in range(length):
            if qids[i] not in self.scores_dict.keys():
                self.scores_dict[qids[i]] = []
            else:
                self.scores_dict[qids[i]].append(scores[i])

            if qids[i] not in self.labels_dict.keys():
                self.labels_dict[qids[i]] = []
            else:
                self.labels_dict[qids[i]].append(labels[i])

        num = 0

        res = dict([[k,0.] for k in metrics])

        for k,v in self.scores_dict.items():
            y_pred = v
            y_true = self.labels_dict[k]
            print(k,v)
            print(y_true,y_pred)
            curr_res = self.eval(y_true=y_true, y_pred=y_pred,
                                      metrics=metrics)
            for k, v in curr_res.items():
                res[k] += v
            num += 1
        print('  '.join(['%s:%f' % (k, v / num) for k, v in res.items()]), ' ...')




    def zipped(self, y_true, y_pred):
        c = list(zip(y_true, y_pred))
        return c

    def eval(self, y_true, y_pred, 
            metrics=['map', 'p@1', 'p@5', 'p@10', 'p@20', 
                'ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20'], k = 20):
        res = {}
        res['map'] = self.map(y_true, y_pred)
        res['mrr'] = self.mrr(y_true, y_pred)
        all_ndcg = self.ndcg(y_true, y_pred, k=k)
        all_precision = self.precision(y_true, y_pred, k=k)
        res.update({'p@%d'%(i+1):all_precision[i] for i in range(k)})
        res.update({'ndcg@%d'%(i+1):all_ndcg[i] for i in range(k)})
        ret = {k:v for k,v in res.items() if k in metrics}
        return ret

    def map(self, y_true, y_pred):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0.
        s = 0.
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                ipos += 1.
                s += ipos / ( 1. + i )
        if ipos == 0:
            return 0.
        else:
            return s / ipos

    def mrr(self,y_true,y_pred):
        s = 0.
        c = list(self.zipped(y_true, y_pred))
        c_s = sorted(c, key=lambda x: x[1], reverse=True)
        for j, (g, p) in enumerate(c_s):
            if g > self.rel_threshold:
                ipos = y_pred.index(p) + 1
                s = 1 / ipos
                break

        return s

    def ndcg(self, y_true, y_pred, k = 20):
        s = 0.
        c = self.zipped(y_true, y_pred)
        c_g = sorted(c, key=lambda x:x[0], reverse=True)
        c_p = sorted(c, key=lambda x:x[1], reverse=True)
        #idcg = [0. for i in range(k)]
        idcg = np.zeros([k], dtype=np.float32)
        dcg = np.zeros([k], dtype=np.float32)
        #dcg = [0. for i in range(k)]
        for i, (g,p) in enumerate(c_g):
            if g > self.rel_threshold:
                idcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for i, (g,p) in enumerate(c_p):
            if g > self.rel_threshold:
                dcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
            if i >= k:
                break
        for idx, v in enumerate(idcg):
            if v == 0.:
                dcg[idx] = 0.
            else:
                dcg[idx] /= v
        return dcg

    def precision(self, y_true, y_pred, k = 20):
        c = self.zipped(y_true, y_pred)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        precision = np.zeros([k], dtype=np.float32) #[0. for i in range(k)]
        for i, (g,p) in enumerate(c):
            if g > self.rel_threshold:
                precision[i:] += 1
            if i >= k:
                break
        precision = [v / (idx + 1) for idx, v in enumerate(precision)]
        return precision


if __name__ == "__main__":
    labels = [0,0,0,1,1,0,1,0,1,0]
    scores = [0.3,0.52,0.48,0.51,0.67,0.98,0.72,0.45,0.89,0.30]
    qids = [0,0,0,0,0,0,0,0,0,0]
    metrics = ["map", "mrr", "p@1", "ndcg@1"]
    rank_eval( qids,labels,scores,metrics)
