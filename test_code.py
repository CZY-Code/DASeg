from libs.dataset.data import VDS
from options import OPTION as opt

def test_dataset():
    data = VDS(transform=opt.test_transforms)
    print(data.length)
    for x in data:
        print(x['support']['img'].shape)
        print(x['support']['mask'].shape)
        print(x['query']['img'].shape)
        print(x['query']['mask'].shape)
        print(x['query']['mask'].dtype)

if __name__ == '__main__':
    test_dataset()