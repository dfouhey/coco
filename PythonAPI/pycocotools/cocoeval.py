__author__ = 'tsungyi,dfouhey'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
from . import bcipr as bciUtils
import scipy.stats as stats
import copy
import md5

#signatures for the replicates that should be generated for datasets with 
#the same size as the COCO minival, val, and test sets. If they don't match, 
#then we raise an exception
_md5replicateSignatures = {5000: '16d20e7a43fcf00dd3626516d817e64d',
                           40504:'6c5db3b9da0a6d203300a29bb8261299',
                           81434:'c842933e485dfc5a275ac7c462459b22'}

def _replicatesToCI(allData, replicates, alpha):
    '''
    Return a (1-alpha) CI using the median-bias corrected method, which offers
    a good tradeoff: the next best is the bias corrected and accelerated one,
    which requires jackknife samples, which entails as many evaluations as
    there are data points.

    See Efron, The jackknife, the bootstrap and other resampling plans, pp 118

    :param allData: results of f(allData)
    :param replicates: a Rx1 matrix containing f(replicate_i) 
    :param alpha: the parameter for the width of the CI
    :return: a 1x2 matrix containing the median-bias corrected CI
    '''
    alpha2 = alpha/2

    z0 = stats.norm.ppf(np.mean(replicates <= allData))
    za = stats.norm.ppf(alpha2)

    #the actual percentiles to take 
    higha = stats.norm.cdf(2*z0-za)
    lowa = stats.norm.cdf(2*z0+za)
    return np.percentile(replicates,[lowa*100,higha*100])


class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # and if p.runBootstrap is true, where B is the number of replicates
    #  APBCI      - [TxBxKxAxM] average precision for every evaluation setting
    #               per-bootstrap replicate
    #  RBCI       - [TxBxKxAxM] max recall for every evaluation setting
    #               per-bootstrap replicate
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Bootstrapping code written by David Fouhey, 2017
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

        
    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))


        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)


        if p.runBootstrap:
            #setup bootstrap
            APBCI = -np.empty((T,p.bootstrapCount,K,A,M),dtype=np.float)
            RBCI = -np.empty((T,p.bootstrapCount,K,A,M),dtype=np.float)

            #save state
            rstate = np.random.get_state()
            np.random.seed(1)

            #map the image ids -> 0, ... N-1 "indexes" for replicates
            listId = list(setI)

            imageIdToIndex = {listId[i]: i for i in range(len(listId))}

            #a bootstrap sample is just reweighting the points; this counts
            bciCounts = np.zeros((p.bootstrapCount,len(listId)))
            for ri in range(p.bootstrapCount):
                #convert the replicate into a count
                samp = np.random.randint(0,len(listId),len(listId))
                bciCounts[ri,:] = np.bincount(samp,minlength=len(listId))

            imageCount = len(listId)
            if imageCount in _md5replicateSignatures and p.verifyBootstrap:
                #if we're evaluating something that looks coco-sized, verify 
                #the replicates are the same
                m = md5.new() 
                m.update("".join(map(lambda f:str(int(f)),bciCounts.ravel().tolist())))
                bootstrapDigest = m.hexdigest()
                if bootstrapDigest != _md5replicateSignatures[imageCount]:
                    raise Exception("Numpy not producing same bootstrap samples; results may not be comparable")

            np.random.set_state(rstate)


        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        indsPR = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(indsPR):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                    
                    if p.runBootstrap:
                        #generate index lists for gtIgnore and dtMatches/dtIgnore
                        imgIds = np.concatenate([np.tile(imageIdToIndex[e['image_id']],e['dtMatches'][:,0:maxDet].shape[1]) for e in E],axis=0)[inds]
                        npigImgIds = np.concatenate([np.tile(imageIdToIndex[e['image_id']],e['gtIgnore'].shape[0]) for e in E],axis=0)[gtIg==0]

                        #call the evaluation code, casting the True/Falses to 1/0s
                        APBCI[:,:,k,a,m],RBCI[:,:,k,a,m] = bciUtils.bcipr(bciCounts,imgIds,npigImgIds,tps.astype(np.float),fps.astype(np.float),p.recThrs)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
        }

        if p.runBootstrap:
            self.eval['APBCI'] = APBCI
            self.eval['RBCI'] = RBCI

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))
        

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this function can *only* be applied on the default parameter setting
        '''
        runBootstrap = self.params.runBootstrap
        numClasses = self.eval['precision'].shape[2]

        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100, classId=None ):
            '''
            Returns a float if bootstrap is false and a 1x3 matrix [mean,low,high] other
            '''
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:s}'
            resultStr = "{:0.3f}" if not self.params.runBootstrap else "{:0.3f} ({:0.3f},{:0.3f})"

            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            cind = slice(0,numClasses) if classId == None else classId

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                if runBootstrap: replicates = self.eval['APBCI'][:,:,cind,aind,mind]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                    if runBootstrap: replicates = replicates[t]
                s = s[:,:,cind,aind,mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if runBootstrap: replicates = self.eval['RBCI'][:,:,cind,aind,mind]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                    if runBootstrap: replicates = replicates[t]
                s = s[:,cind,aind,mind]

            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])

            #make result string
            if runBootstrap:
                #make the replicates the first dimension
                replicates = np.moveaxis(replicates,1,0)

                #ignore invalid samples; this is very rare, and there's no good solution; this
                #imitates the standard eval code
                replicates = np.nanmean(replicates,axis=tuple(range(1,len(replicates.shape))))
                replicates = replicates[np.isnan(replicates)==False]
                bci = _replicatesToCI(mean_s, replicates, p.bootstrapAlpha)
                result = resultStr.format(mean_s,bci[0],bci[1])
            else:
                result = resultStr.format(mean_s)

            if classId is None:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, result))

            return np.array([mean_s,bci[0],bci[1]]) if runBootstrap else mean_s
        def _summarizeDets():

            stats = np.zeros((12+numClasses,3)) if runBootstrap else np.zeros((12,1))

            stats[0,:] = _summarize(1)
            stats[1,:] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2,:] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3,:] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4,:] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5,:] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6,:] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7,:] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8,:] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9,:] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10,:] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11,:] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

            if runBootstrap:
                for k in range(numClasses):
                    stats[12+k,:] = _summarize(1,classId=k)

            stats = np.squeeze(stats)
            return stats
        def _summarizeKps():
            stats = np.zeros((10,3)) if runBootstrap else np.zeros((10,1))
            stats[0,:] = _summarize(1, maxDets=20)
            stats[1,:] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2,:] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3,:] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4,:] = _summarize(1, maxDets=20, areaRng='large')
            stats[5,:] = _summarize(0, maxDets=20)
            stats[6,:] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7,:] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8,:] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9,:] = _summarize(0, maxDets=20, areaRng='large')

            stats = np.squeeze(stats)
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __sub__(self,other):
        '''
        Take two cocoEval instances and return the one representing the 
        difference of the results. This is primarily useful for computing
        paired bootstraps
        :param other: another cocoEval instance
        :return: a new cocoEval instance representing the difference
        '''
        diffEval = COCOeval(None,None,self.params.iouType)
        diffEval.params = self.params
        if not self.eval or not other.eval:
            raise Exception("Please run accumulate() first on both operands")

        for k in ['recall','precision','APBCI','RBCI']:
            if k in self.eval and k in other.eval:
                diffEval.eval[k] = self.eval[k]-other.eval[k]

        return diffEval

    def rank(self,others):
        """
        Compute bootstrapped ranking information for a list of evals. The 
        metrics that are used are the standard COCO ones, plus one per-class
        using the main competition criterion.

        :param others: a list of N cocoEval instances 
        :return: a tuple (A,B) 
        (A) A Nx3xK array M where M[i,:,k] is the average and CI of method i's 
            rank on metric k

        (B) a NxNxK array M where M[i,j,k] is the frequency with which method i 
            is better than method j according to metric k

        """
        types = [o.params.iouType for o in others]
        if len(set(types)) != 1:
            raise Exception("Need to be all the same type")

        if not all((o.eval)):
            raise Exception("Please run accumulate() on all the arguments.")
        
        bootstrappedSize = [0 if 'APBCI' not in o.eval else o.eval['APBCI'].shape[1]]
        if min(bootstrappedSize) == 0 or len(set(bootstrappedSize)) != 1:
            raise Exception("Please run accumulate() with bootstrapping on and identical settings")
        
        #construct the n evaluation criteria + classes in an extensible way
        #evalFn = [AP,R] in the standard format -> column with as many rows as replicates
        numClasses = others[0].eval['APBCI'].shape[2]

        if types[0] in ["segm","bbox"]:
            evalFunctions = [ \
                lambda AP,R: np.nanmean(AP[:,:,:,0,-1],axis=(0,2)),
                lambda AP,R: np.nanmean(AP[0,:,:,0,-1],axis=(1)),
                lambda AP,R: np.nanmean(AP[5,:,:,0,-1],axis=(1)),
                lambda AP,R: np.nanmean(AP[:,:,:,1,-1],axis=(0,2)),
                lambda AP,R: np.nanmean(AP[:,:,:,2,-1],axis=(0,2)),
                lambda AP,R: np.nanmean(AP[:,:,:,3,-1],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,0,0],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,0,1],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,0,2],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,1,2],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,2,2],axis=(0,2)),
                lambda AP,R: np.nanmean(R[:,:,:,3,2],axis=(0,2))]

            evfAP = lambda c: (lambda AP,R: np.nanmean(AP[:,:,c,0,-1],axis=0))
            for i in range(numClasses):
                evalFunctions.append(evfAP(i))

        else:
            evalFunctions = [ \
                    lambda AP,R: np.nanmean(AP[:,:,:,0,0],axis=(0,2)),
                    lambda AP,R: np.nanmean(AP[0,:,:,0,0],axis=(1)),
                    lambda AP,R: np.nanmean(AP[5,:,:,0,0],axis=(1)),
                    lambda AP,R: np.nanmean(AP[:,:,:,1,0],axis=(0,2)),
                    lambda AP,R: np.nanmean(AP[:,:,:,2,0],axis=(0,2)),
                    lambda AP,R: np.nanmean(R[:,:,:,0,0],axis=(0,2)),
                    lambda AP,R: np.nanmean(R[0,:,:,0,0],axis=(1)),
                    lambda AP,R: np.nanmean(R[5,:,:,0,0],axis=(1)),
                    lambda AP,R: np.nanmean(R[:,:,:,1,0],axis=(0,2)),
                    lambda AP,R: np.nanmean(R[:,:,:,2,0],axis=(0,2))]

        numReplicates = others[0].eval['APBCI'].shape[1]
        numInstances = len(others)
        numEvals = len(evalFunctions)

        replicateStats = np.zeros((numReplicates,numInstances))
        ranks = np.zeros((numReplicates,numInstances))

        outperformMatrix = np.zeros((numInstances,numInstances,numEvals))
        rankCI = np.zeros((numInstances,3,numEvals))

        for evi,evf in enumerate(evalFunctions):
            for oi,o in enumerate(others):
                replicateStats[:,oi] = evf(o.eval['APBCI'],o.eval['RBCI'])

            for oi in range(len(others)):
                for oj in range(len(others)):
                    outperformMatrix[oi,oj,evi] = np.mean(replicateStats[:,oi]>replicateStats[:,oj])

            for bci in range(numReplicates):
                ranks[bci,:] = stats.rankdata(-replicateStats[bci,:],method='min')

            for oi in range(len(others)): 
                rankCI[oi,0,evi] = np.mean(ranks[:,oi])
                #use simple percentile method; the bias correction misbehaves 
                rankCI[oi,1:,evi] = np.percentile(ranks[:,oi],[100*(self.params.bootstrapAlpha/2),100*(1-self.params.bootstrapAlpha/2)])

        return rankCI, outperformMatrix

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1
        self.runBootstrap = 0
        self.bootstrapCount = 1000
        self.bootstrapAlpha = 0.05
        self.verifyBootstrap = True

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.runBootstrap = 0
        self.bootstrapCount = 1000
        self.bootstrapAlpha = 0.05
        self.verifyBootstrap = True


    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
