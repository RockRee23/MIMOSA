import torch   
def evaluation(model, ub_loader, topks):
    model.eval()
    
    tmp_metrics = {}
    for topk in topks:
        tmp_metrics[topk] = {'recall':[0, 0], 'ndcg':[0, 0]}
    for u_idx, ground_truth_ub, train_mask_ub in ub_loader:
        preds = model.bnd_prediction(u_idx)
        preds -= 1e8 * train_mask_ub.cuda()
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_ub, preds, topks)
        
    metrics = {}
    for topk in topks:
        metrics[topk]={}
        for m, res in tmp_metrics[topk].items():
            metrics[topk][m] = res[0] / res[1]
    return metrics

def get_metrics(metrics, grd, pred, topks):
    tmp = {}
    for topk in topks:
        tmp[topk]={'recall':0, 'ndcg':0}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice)+torch.arange(pred.shape[0], dtype=torch.long).view(-1, 1).cuda()
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp[topk]['recall'] = get_recall(pred, grd, is_hit, topk)
        tmp[topk]['ndcg'] = get_ndcg(pred, grd, is_hit, topk)
    for topk in topks:
        for m, res in tmp[topk].items():
            metrics[topk][m][0] += tmp[topk][m][0]
            metrics[topk][m][1] += tmp[topk][m][1]
    return metrics
    
def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt / (num_pos + epsilon)).sum().item()#need hit_cnt/(num_pos+epsilon), num_pos

    return [nomina, denorm]
    
def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk):
        index = torch.arange(2, topk + 2, dtype=torch.float)
        index.cuda()
        hit = hit / torch.log2(index)
        return hit.sum(-1)

    def IDCG(num_pos, topk):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk)

    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk)

    idcg = IDCGs[num_pos]
    idcg.cuda()
    ndcg = dcg / idcg

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()#need ndcg, num_pos

    return [nomina, denorm]
