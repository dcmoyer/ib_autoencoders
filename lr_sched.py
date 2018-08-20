# LR SCHEDULER
def iwae_lr_sched(epoch):
    ''' based on 3280 epochs '''
    i_sum = 0
    i = 0
    i_sum = i_sum + 3**i 
    while epoch+1 > i_sum:
        i = i+1
        i_sum = i_sum + 3**i
    lr = .001*10**(-i/7.0)
    print(epoch, ': lr = ', lr)
    return lr