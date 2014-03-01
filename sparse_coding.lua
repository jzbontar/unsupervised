require 'sys'
require '1_data'
require 'image'
require 'nn'
require 'optim'
require 'unsup'
require 'torch_datasets'
require 'fista'
require 'cunn'
require 'cutorch'
require 'jzt'

inputsize = 28
nfiltersout = 512
batch_size = 128

-- torch.setdefaulttensortype('torch.CudaTensor')

-- datafile = 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.ascii'
-- filename = paths.basename(datafile)
-- if not paths.filep(filename) then
--    os.execute('wget ' .. datafile)
-- end
-- dataset = getdata(filename, inputsize)

X_tr, y_tr, X_te, y_te = torch_datasets.mnist()
mean = X_tr:mean()
std = X_tr:std()
X_tr = X_tr:add(-mean):div(std)

fista_mse = nn.MSECriterion()
fista_mse.sizeAverage = false
sgd_mse = nn.MSECriterion()
sgd_mse.sizeAverage = false

inputSize = inputsize * inputsize
outputSize = nfiltersout

decoder = nn.Linear(outputSize, inputSize)
decoder.bias:fill(0)

code = torch.Tensor(batch_size, outputSize)
fista_params = {}
fista_params.L = 256
fista_params.maxiter = 50

sgd_params = {}
sgd_params.learningRate = 2e-3
sgd_params.learningRateDecay = 1e-5


-- float
X_tr = X_tr:cuda()
fista_mse = fista_mse:cuda()
decoder = decoder:cuda()
code = code:cuda()

dec, grad_dec = decoder:getParameters()

sys.tic()
iter = -1
for epoch = 1,10 do
   for t = 1,60000 - batch_size, batch_size do
      iter = iter + 1
      input = X_tr:narrow(1, t, batch_size)

      -- 1. update code (fista)
      function f_fista(x, mode)
         decoder:updateOutput(x)
         fista_mse:updateOutput(decoder.output, input)
         if mode and mode:match('dx') then
            fista_mse:updateGradInput(decoder.output, input)
            decoder:updateGradInput(x, fista_mse.gradInput)
            return fista_mse.output, decoder.gradInput
         end
         return fista_mse.output, nil
      end

      function prox_fista(x, L)
         jzt.shrink(x, 1 / L, x)
      end

      code:fill(0)
      f = fista(f_fista, nil, prox_fista, code, fista_params)
      if f > 1e6 then
         print('f is really big')
      end

      -- 2. update encoder and decoder
      decoder:zeroGradParameters()
      decoder:accGradParameters(code, fista_mse.gradInput)
      decoder.gradBias:fill(0)

      optim.sgd(function(x) return 0, grad_dec end, dec, sgdconf)

      -- normalize the dictionary
      jzt.div_mat_vect(decoder.weight, decoder.weight:norm(2, 1), decoder.weight, 1)

      if math.fmod(iter, 100) == 0 then
         -- get weights
         dweight = decoder.weight
         dweight = dweight:transpose(1,2):unfold(2,inputsize,inputsize)

         -- render filters
         dd = image.toDisplayTensor{input=dweight,
                                    padding=2,
                                    nrow=math.floor(math.sqrt(nfiltersout)),
                                    symmetric=true}

         image.savePNG(string.format('img/d_%02d_%010d.png', epoch, t), dd)
         print(epoch, t, sys.toc())
         sys.tic()
      end
   end
end
