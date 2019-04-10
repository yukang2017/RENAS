"""
learning rate scheduler, which adaptive changes the learning rate based on the
progress
"""
import logging
from mxnet.lr_scheduler import * 

def _var_dump(obj):
  '''return a printable representation of an object for debugging'''
  newobj=obj
  if '__dict__' in dir(obj):
    newobj=obj.__dict__
    if ' object at ' in str(obj) and not newobj.has_key('__type__'):
      newobj['__type__']=str(obj)
    for attr in newobj:
      newobj[attr]=_var_dump(newobj[attr])
  return newobj

class ComposedLRScheduler(LRScheduler):
    '''
    @brief A learning rate scheduler that contains other schedulers. 
           If a inside learning rate scheduler exceed its life period, 
           the next scheduler will be applied.  
    '''
    def __init__(self, base_lr=0.01):
        super(ComposedLRScheduler, self).__init__(base_lr)
        self.scheduler_list = []
        self.scheduler_start = []
        self.changing = False

    def add(self, scheduler, num_update_for_activation):
        '''
        @brief Add a learning rate scheduler.
        
        Parameters
        ----------
        scheduler : LRScheduler
            The scheduler to add
            
        num_update_for_activation : int
            The start num_update to activate this scheduler.
        '''
        assert isinstance(scheduler, LRScheduler)
        assert isinstance(num_update_for_activation, int)
        assert num_update_for_activation >= 0
        self.scheduler_list.append(scheduler)
        self.scheduler_start.append(num_update_for_activation)
        if len(self.scheduler_start) == 1 and \
            num_update_for_activation > 0:
            self.scheduler_start[0] = 0

    def __call__(self, num_update):
        chosen_id = 0
        # find the last scheduler that can be activated
        for i in range(len(self.scheduler_start)):
            if num_update >=  self.scheduler_start[i]:
                chosen_id = i
        if num_update == self.scheduler_start[chosen_id]:
            if not self.changing and chosen_id > 0:
                logging.info("change lr_scheduler to %s"%( _var_dump(self.scheduler_list[chosen_id])))
                self.changing = True
            self.scheduler_list[chosen_id].base_lr = self.base_lr
        else:
            self.changing = False
        self.base_lr = self.scheduler_list[chosen_id](
            num_update - self.scheduler_start[chosen_id])
        return self.base_lr
        
class InvScheduler(LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr *  pow(Dtype(1) + this->param_.gamma() * this->iter_, - this->param_.power());

    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, gamma = 0.0001, power = 0.75, stop_factor_lr=1e-8):
        super(InvScheduler, self).__init__()
        if gamma < 0:
            raise ValueError("Schedule gamma must be greater than 0")
        if power < 0:
            raise ValueError("power must be no more than 0 to make lr reduce")
        self.gamma = gamma
        self.power = power
        self.stop_factor_lr = stop_factor_lr
        self.count = 0
        self.start_lr = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        self.count = num_update
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        self.base_lr = self.start_lr * pow(1 + self.gamma * num_update, -self.power);
        if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr 
        return self.base_lr
     


        
class PolyScheduler(LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

     base_lr * (1 - num_update/max_num_update) ^ (power)

    Parameters
    ----------
    max_num_update : int default = 300000
        max num_update of training. 
    power : float  default = 1.4
        The power to change lr.  
    stop_factor_lr : float, default = 1e-8
        Stop updating the learning rate if it is less than this value.
    """
    def __init__(self, max_num_update = 300000, 
                 power = 1.4, stop_factor_lr=1e-8):
        super(PolyScheduler, self).__init__()
        if max_num_update < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if power < 0:
            raise ValueError("power must be no more than 0 to make lr reduce")
        self.max_num_update = max_num_update
        self.power = power
        self.stop_factor_lr = stop_factor_lr 
        self.start_lr = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """ 
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        
        if num_update < self.max_num_update:
            self.base_lr = self.start_lr * pow(1 - float(num_update)/self.max_num_update, 
                                               self.power);
        else:
            self.base_lr = self.stop_factor_lr 
            
        if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr 
        return self.base_lr


class LinearWarmUpScheduler(LRScheduler):
    """Reduce learning rate in factor

    Assume the weight has been updated by n times, then the learning rate will
    be

    self.base_lr = self.start_lr + \
                float(self.lr_end - self.start_lr)/self.peroid_size * num_update 

    Parameters
    ---------- 
    """
    def __init__(self, peroid_size, lr_end = 0.1):
        super(LinearWarmUpScheduler, self).__init__()
        if lr_end < 0:
            raise ValueError("Schedule lr_end must be greater than 0") 
        if peroid_size < 0:
            raise ValueError("Schedule peroid_size must be greater than 0") 
        self.lr_end = lr_end
        self.peroid_size = peroid_size 
        self.start_lr = -1

    def __call__(self, num_update):  
        if self.start_lr < 0:
            self.start_lr = self.base_lr
        
        if num_update < self.peroid_size:
            self.base_lr = self.start_lr + \
                float(self.lr_end - self.start_lr)/self.peroid_size * num_update 
        else:
            self.base_lr = self.lr_end  
        return self.base_lr

class WarmupMultiFactorScheduler(LRScheduler):

    def __init__(self, step, factor=1, warmup=False, 
            warmup_linear = True, warmup_lr=0, warmup_step=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
            if factor > 1.0:
                raise ValueError("Factor must be no more than 1 to make lr reduce")
        assert warmup_step <= step[0]
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_linear = warmup_linear
        self.warmup_lr = warmup_lr if self.warmup else 0
        self.warmup_start_lr= warmup_lr if self.warmup else 0
        self.warmup_step = warmup_step if self.warmup else 0
        self.warmup_end_lr=warmup_lr if self.warmup else 0
        self.start_lr=-1
        

    def __call__(self, num_update):
        if self.start_lr==-1:
           self.start_lr=self.base_lr
           if self.warmup and self.warmup_linear:
                self.warmup_end_lr=self.base_lr
        if self.warmup and num_update < self.warmup_step:
            if not self.warmup_linear:
                self.base_lr = self.warmup_lr
                return self.base_lr
            else:
                self.base_lr = self.warmup_start_lr +\
                                 float(self.warmup_end_lr-self.warmup_start_lr)/self.warmup_step*num_update
                return self.base_lr
        while self.cur_step_ind <= len(self.step)-1:
            self.base_lr=self.start_lr
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind+=1
                self.base_lr*=self.factor
                self.start_lr=self.base_lr
                logging.info("Update[%d]: Change learning rate to %0.5e",
                                                    num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr





class WarmupPolyScheduler(LRScheduler):

    def __init__(self,power =1.0, max_num_update =30000,stop_factor_lr=1e-8,warmup=False, 
            warmup_linear = False, warmup_lr=0, warmup_step=0):
        super(WarmupPolyScheduler, self).__init__()
        if max_num_update < 0:
            raise ValueError("Schedule max_num_update must be greater than 0")
        if power < 0:
            raise ValueError("power must be no more than 0 to make lr reduce")
        assert warmup_step <max_num_update
        self.max_num_update = max_num_update
        self.power = power
        self.stop_factor_lr = stop_factor_lr
        self.warmup = warmup
        self.warmup_linear = warmup_linear
        self.warmup_lr = warmup_lr if self.warmup else 0
        self.warmup_start_lr = warmup_lr if self.warmup else 0
        self.warmup_end_lr = warmup_lr if self.warmup else 0
        self.warmup_step = warmup_step if self.warmup_lr else 0
        self.start_lr = -1

    def __call__(self, num_update):

        if self.start_lr==-1:
           self.start_lr=self.base_lr
           if self.warmup and self.warmup_linear:
                self.warmup_end_lr=self.base_lr
        if self.warmup and num_update < self.warmup_step:
            if not self.warmup_linear:
                #self.base_lr = self.warmup_lr
                #print self.warmup_linear
                self.base_lr=self.warmup_lr
                return self.base_lr
            else:
                self.base_lr = self.warmup_start_lr + \
                                 float(self.warmup_end_lr-self.warmup_start_lr)/self.warmup_step*num_update
                #print self.warmup_lr
                return self.base_lr
        
        if num_update < self.max_num_update:
            self.base_lr = self.start_lr * pow(1 - float(num_update-self.warmup_step)/(self.max_num_update-self.warmup_step), 
                                               self.power)
        else:
            self.base_lr = self.stop_factor_lr 
            
        if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr 
        return self.base_lr











