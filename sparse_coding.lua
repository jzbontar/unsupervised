require 'sys'
require '1_data'
require 'image'
require 'nn'
require 'optim'
require 'unsup'
require 'torch_datasets'

inputsize = 28
nfiltersout = 256
batch_size = 128

-- datafile = 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.ascii'
-- filename = paths.basename(datafile)
-- if not paths.filep(filename) then
--    os.execute('wget ' .. datafile)
-- end
-- dataset = getdata(filename, inputsize)

X_tr, y_tr, X_te, y_te = torch_datasets.mnist()
X_tr = X_tr:double()
mean = X_tr:mean()
std = X_tr:std()
X_tr = X_tr:add(-mean):div(std)

fista_mse = nn.MSECriterion()
fista_mse.sizeAverage = false
sgd_mse = nn.MSECriterion()
sgd_mse.sizeAverage = false
l1 = nn.L1Cost()

inputSize = inputsize * inputsize
outputSize = nfiltersout

decoder = nn.Linear(outputSize, inputSize)
decoder.bias:fill(0)

code = torch.Tensor(batch_size, outputSize)
fista_params = {}
fista_params.L = 0.1
fista_params.Lstep = 1.5
fista_params.maxiter = 50
fista_params.maxline = 20
fista_params.errthres = 1e-4
fista_params.doFistaUpdate = true

sgd_params = {}
sgd_params.learningRate = 2e-3
sgd_params.learningRateDecay = 1e-5

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

      function g_fista(x, mode)
         l1:updateOutput(x)
         if mode and mode:match('dx') then
            l1:updateGradInput(x)
            return l1.output, l1.gradInput
         end
         return l1.output, nil
      end

      function prox_fista(x, L)
         x:shrinkage(1 / L)
      end

      code:fill(0)
      optim.FistaLS(f_fista, g_fista, prox_fista, code, fista_params)

      -- 2. update encoder and decoder
      decoder:zeroGradParameters()
      decoder:accGradParameters(code, fista_mse.gradInput)
      decoder.gradBias:fill(0)

      optim.sgd(function(x) return 0, grad_dec end, dec, sgdconf)

      -- normalize the dictionary
      w = decoder.weight
      for i=1,w:size(2) do
         w:select(2,i):div(w:select(2,i):norm()+1e-12)
      end

      if math.fmod(iter, 10) == 0 then
         -- get weights
         dweight = decoder.weight
         dweight = dweight:transpose(1,2):unfold(2,inputsize,inputsize)

         -- render filters
         dd = image.toDisplayTensor{input=dweight,
                                    padding=2,
                                    nrow=math.floor(math.sqrt(nfiltersout)),
                                    symmetric=true}

         image.savePNG(string.format('img/d_%010d.png', t), dd)
         print(t, sys.toc())
         sys.tic()
      end
   end
end
