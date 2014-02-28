require 'sys'
require '1_data'
require 'image'
require 'nn'
require 'optim'
require 'unsup'

inputsize = 12
nfiltersout = 256
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
-- 
-- torch.save('dataset.bin', dataset_cache)
-- dataset = torch.load('dataset.bin')

fista_mse = nn.MSECriterion()
fista_mse.sizeAverage = false
sgd_mse = nn.MSECriterion()
sgd_mse.sizeAverage = false
l1 = nn.L1Cost()

inputSize = inputsize * inputsize
outputSize = nfiltersout

encoder = nn.Sequential()
encoder:add(nn.Linear(inputSize, outputSize))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(outputSize))

decoder = nn.Linear(outputSize, inputSize)
decoder.bias:fill(0)

code = torch.Tensor(outputSize)
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

tmp = nn.Sequential()
tmp:add(encoder)
tmp:add(decoder)
encdec, grad_encdec = tmp:getParameters()

sys.tic()
for t = 1,maxiter do
   sample = dataset[t]
   input = sample[1]:clone()
   target = sample[2]:clone()

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
   encoder:zeroGradParameters()
   encoder:updateOutput(input)
   sgd_mse:updateOutput(encoder.output, code)
   sgd_mse:updateGradInput(encoder.output, code)
   encoder:updateGradInput(input, sgd_mse.gradInput)
   encoder:accGradParameters(input, sgd_mse.gradInput)

   decoder:zeroGradParameters()
   decoder:accGradParameters(code, fista_mse.gradInput)
   decoder.gradBias:fill(0)

   optim.sgd(function(x) return 0, grad_encdec end, encdec, sgdconf)

   -- normalize the dictionary
   w = decoder.weight
   for i=1,w:size(2) do
      w:select(2,i):div(w:select(2,i):norm()+1e-12)
   end

   if math.fmod(t, 5000) == 0 then
      -- get weights
      eweight = encoder.modules[1].weight
      dweight = decoder.weight

      dweight = dweight:transpose(1,2):unfold(2,inputsize,inputsize)
      eweight = eweight:unfold(2,inputsize,inputsize)

      -- render filters
      dd = image.toDisplayTensor{input=dweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(nfiltersout)),
                                 symmetric=true}
      de = image.toDisplayTensor{input=eweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(nfiltersout)),
                                 symmetric=true}

      -- live display
      image.savePNG(string.format('img/d_%010d.png', t), dd)
      image.savePNG(string.format('img/e_%010d.png', t), de)
      print(sys.toc())
      sys.tic()
   end
end
