import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    #def coordinate()

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        #4,5,3,50,50
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        errors = [0 for _ in range(self.update_step + 1)]
        for i in range(task_num):
            y=torch.Tensor(y_qry[i].shape[0],2)
            '''
            y[:,0]=y_qry[i]//15*1.2
            y[:,1]=y_qry[i]%15*0.6
            y[torch.nonzero(y_qry[i]>14),0]+=0.6
            y[torch.nonzero(y_qry[i]>=75),0]+=0.6'''
            
            y[:,0]=y_qry[i]//6*0.6
            y[:,1]=y_qry[i]%6*0.6
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
        
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
                prob=torch.topk(inter_pred_q,5)[0]
                inter_pred_q=torch.topk(inter_pred_q,5)[1]
                pred_q=torch.Tensor(y_qry[i].shape[0],2)
                
                pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                '''
                pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                indd=inter_pred_q.clone() #label of inter_pred_q
                inter_pred_q=inter_pred_q//15*1.2
                inter_pred_q[torch.where(indd>14)]+=0.6
                inter_pred_q[torch.where(indd>=75)]+=0.6
                pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
                error=torch.norm(pred_q-y,dim=1).mean()
                errors[0]+=error
                '''
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct'''
            
            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                '''
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct'''
                inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
                prob=torch.topk(inter_pred_q,5)[0]
                inter_pred_q=torch.topk(inter_pred_q,5)[1]
                pred_q=torch.Tensor(y_qry[i].shape[0],2)
                
                pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                '''
                pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                indd=inter_pred_q.clone() #label of inter_pred_q
                inter_pred_q=inter_pred_q//15*1.2
                inter_pred_q[torch.where(indd>14)]+=0.6
                inter_pred_q[torch.where(indd>=75)]+=0.6
                pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
                error=torch.norm(pred_q-y,dim=1).mean()
                errors[1]+=error

            

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    '''
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct'''
                    inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
                    prob=torch.topk(inter_pred_q,5)[0]
                    inter_pred_q=torch.topk(inter_pred_q,5)[1]
                    pred_q=torch.Tensor(y_qry[i].shape[0],2)
                    
                    pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                    pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                    '''
                    pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                    indd=inter_pred_q.clone() #label of inter_pred_q
                    inter_pred_q=inter_pred_q//15*1.2
                    inter_pred_q[torch.where(indd>14)]+=0.6
                    inter_pred_q[torch.where(indd>=75)]+=0.6
                    pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
                    error=torch.norm(pred_q-y,dim=1).mean()
                    errors[k+1]+=error



        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        #accs = np.array(corrects) / (querysz * task_num)
        accs = np.array(errors) / (task_num)
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry,x_test,y_test):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        errors = [0 for _ in range(self.update_step_test + 1)]
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        y=torch.Tensor(y_qry.shape[0],2)
        '''
        y[:,0]=y_qry//15*1.2
        y[:,1]=y_qry%15*0.6
        y[torch.nonzero(y_qry>14),0]+=0.6
        y[torch.nonzero(y_qry>=75),0]+=0.6'''
        
        y[:,0]=y_qry//6*0.6
        y[:,1]=y_qry%6*0.6

        y_=torch.Tensor(y_test.shape[0],2)
        '''
        y_[:,0]=y_test//15*1.2
        y_[:,1]=y_test%15*0.6
        y_[torch.nonzero(y_test>14),0]+=0.6
        y_[torch.nonzero(y_test>=75),0]+=0.6'''
        
        y_[:,0]=y_test//6*0.6
        y_[:,1]=y_test%6*0.6

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
            prob=torch.topk(inter_pred_q,5)[0]
            inter_pred_q=torch.topk(inter_pred_q,5)[1]
            pred_q=torch.Tensor(y_qry.shape[0],2)
            
            pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            '''
            pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            indd=inter_pred_q.clone() #label of inter_pred_q
            inter_pred_q=inter_pred_q//15*1.2
            inter_pred_q[torch.where(indd>14)]+=0.6
            inter_pred_q[torch.where(indd>=75)]+=0.6
            pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
            #print('33333333333333333333333')
            #print(torch.norm(pred_q-y,dim=1))
            error=torch.norm(pred_q-y,dim=1).mean()
            errors[0]+=error

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
            prob=torch.topk(inter_pred_q,5)[0]
            inter_pred_q=torch.topk(inter_pred_q,5)[1]
            pred_q=torch.Tensor(y_qry.shape[0],2)
            
            pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            '''
            pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
            indd=inter_pred_q.clone() #label of inter_pred_q
            inter_pred_q=inter_pred_q//15*1.2
            inter_pred_q[torch.where(indd>14)]+=0.6
            inter_pred_q[torch.where(indd>=75)]+=0.6
            pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
            error=torch.norm(pred_q-y,dim=1).mean()
            errors[1]+=error

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            #print(x_spt.shape)
            #print('444444444444444444')
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            

            with torch.no_grad():
                inter_pred_q = F.softmax(logits_q, dim=1)#.argmax(dim=1)
                
                prob=torch.topk(inter_pred_q,5)[0]
                
                #print(prob.shape) #360*5
                inter_pred_q=torch.topk(inter_pred_q,5)[1]
                
                #print("2222222222222222")
                #print(y_qry.shape[0])
                pred_q=torch.Tensor(y_qry.shape[0],2)
                
                pred_q[:,0]=((inter_pred_q//6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                pred_q[:,1]=((inter_pred_q%6*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                '''
                pred_q[:,1]=((inter_pred_q%15*0.6*prob).sum(axis=1))/prob.sum(axis=1)
                indd=inter_pred_q.clone() #label of inter_pred_q
                inter_pred_q=inter_pred_q//15*1.2
                inter_pred_q[torch.where(indd>14)]+=0.6
                inter_pred_q[torch.where(indd>=75)]+=0.6
                pred_q[:,0]=((inter_pred_q*prob).sum(axis=1))/prob.sum(axis=1)'''
                error=torch.norm(pred_q-y,dim=1).mean()
                errors[k+1]+=error
        #print(x_test.shape)
        
        #print('333333333333333333')
        
        logits_test = net(x_test, fast_weights, bn_training=False)
        inter_pred_test = F.softmax(logits_test, dim=1)#.argmax(dim=1)
        inter_pred_test=torch.topk(inter_pred_test,5)[1]
        prob_test=torch.topk(inter_pred_test,5)[0]
        pred_test=torch.Tensor(y_.shape[0],2)
        
        pred_test[:,0]=((inter_pred_test//6*0.6*prob_test).sum(axis=1))/prob_test.sum(axis=1)
        pred_test[:,1]=((inter_pred_test%6*0.6*prob_test).sum(axis=1))/prob_test.sum(axis=1)
        '''
        pred_test[:,1]=((inter_pred_test%15*0.6*prob_test).sum(axis=1))/prob_test.sum(axis=1)
        indd=inter_pred_test.clone() #label of inter_pred_q
        inter_pred_test=inter_pred_test//15*1.2
        inter_pred_test[torch.where(indd>14)]+=0.6
        inter_pred_test[torch.where(indd>=75)]+=0.6
        pred_test[:,0]=((inter_pred_test*prob_test).sum(axis=1))/prob_test.sum(axis=1)'''
        '''
        print('000000000000')
        print(pred_test)
        print(y_)'''
        error_test=pred_test-y_
        error_test_mean=torch.norm(error_test,dim=1).mean()

        del net

        #accs = np.array(corrects) / querysz
        accs = errors
        accs_test = error_test_mean
        return (accs, accs_test)



def main():
    pass


if __name__ == '__main__':
    main()
