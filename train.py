# test pytorch implementation
from argparse import ArgumentParser

from torchtext import data
from torchtext import datasets

from nse import NSE

# replace with args after i get it working
parser = ArgumentParser(description='nse pytorch')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=1) #28)
parser.add_argument('--dim_size', type=int, default=100)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--train_embed', action='store_false', dest='fix_embeds')
parser.add_argument('--word_vectors', type=str, default='glove.42b')
parser.add_argument('--dropout', type=float, dest='p', default=0.3)

args = parser.parse_args()

config = args

inputs = data.Field(lower=True)
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev,test), batch_size=config.batch_size, device=config.gpu)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)


model = NSE(config)

for batch_idx, batch in enumerate(train_iter):
    model.train()
    answer = model(batch)


