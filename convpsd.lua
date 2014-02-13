require 'sys'
require '1_data'
-- require 'gfx.go'
require 'nn'
require 'optim'
require 'unsup'

inputsize = 25
nfiltersout = 16
maxiter = 1000000
datafile = 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.ascii'
 
filename = paths.basename(datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. datafile)
end
dataset = getdata(filename, inputsize)

-- dataset_cache = {}
-- for t = 1,10 do
--    sample = dataset[t]
--    input = sample[1]:clone()
--    target = sample[2]:clone()
--    dataset_cache[t] = {input, target}
-- end

torch.save('dataset.bin', dataset_cache)
dataset = torch.load('dataset.bin')

fista_mse = nn.MSECriterion()
fista_mse.sizeAverage = false
sgd_mse = nn.MSECriterion()
sgd_mse.sizeAverage = false
l1 = nn.L1Cost()
