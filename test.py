from evaluation import Evaluator

def test_model(models, test_loaders, save_pth):
    
    evaluator = Tester(test_loaders)
    res = {}
    total_avg = []
    
    for client, model in models.items():
        dice = evaluator.eval(model, client)
        res[client] = dice
        meanDice = np.mean(dice, axis=1)
        avg = np.round(np.mean(meanDice),4)
        std = np.round(np.std(meanDice),4)
        total_avg.append(meanDice)
        print(client+': {}, {}'.format(avg, std))
        
    print('Total Average: {}, {}'.format(np.mean(total_avg),np.std(total_avg)))
    total_avg = np.concatenate(total_avg)
    np.save(save_pth+'results.npy', res, allow_pickle=True)
        
    
class Tester(Evaluator):
    
    def eval(self, model,client):
        
        num_cls = 4
        total_overlap = np.zeros((1, num_cls, 5))
        res = {}
        model.eval()
        for vali_batch in self.vali_loaders[client]:

            imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            labs = vali_batch['seg']

            output= model(imgs)
 
            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)
            
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            
            overlap_result, _ = Hausdorff_compute(truearg, labs, num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)
            
            #del input, truearg0, truemax
        
        dice = total_overlap[1:,1:,1]
        
        return dice

