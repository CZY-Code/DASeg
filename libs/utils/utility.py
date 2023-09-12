import os
import torch
import shutil

def save_checkpoint(state, epoch, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename +str(epoch)+ '.pth.tar')
    torch.save(state, filepath)
    print('==> save model at {}'.format(filepath))
    if is_best:
        cpy_file = os.path.join(checkpoint, filename+'_model_best.pth.tar')
        shutil.copyfile(filepath, cpy_file)
        print('==> save best model at {}'.format(cpy_file))