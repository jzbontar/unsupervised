require 'sys'
require '1_data'
require 'gfx.go'
require 'xlua'
require 'nn'
require 'optim'
require 'unsup'

inputsize = 25
kernelsize = 16
nfiltersin = 1
nfiltersout = 9
maxiter = 1000000
datafile = 'http://data.neuflow.org/data/tr-berkeley-N5K-M56x56-lcn.ascii'
 
filename = paths.basename(datafile)
if not paths.filep(filename) then
   os.execute('wget ' .. datafile)
end
dataset = getdata(filename, inputsize)
dataset:conv()
-- 
-- dataset_cache = {}
-- for t = 1,10 do
--    sample = dataset[t]
--    input = sample[1]:clone()
--    target = sample[2]:clone()
--    dataset_cache[t] = {input, target}
-- end
-- torch.save('dataset.bin', dataset_cache)

-- dataset = torch.load('dataset.bin')

kw, kh = kernelsize, kernelsize
iw, ih = inputsize, inputsize

tt = torch.Tensor(nfiltersin, ih, iw)
utt = tt:unfold(2, kh, 1):unfold(3, kw, 1)
tt:zero()
utt:add(1)
tt:div(tt:max())
fista_mse = nn.WeightedMSECriterion(tt)
fista_mse.sizeAverage = false

sgd_mse = nn.MSECriterion()
sgd_mse.sizeAverage = false

l1 = nn.L1Cost()

encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(nfiltersin, nfiltersout, kw, kh, 1, 1))
encoder:add(nn.Tanh())
encoder:add(nn.Diag(nfiltersout))

decoder = nn.SpatialFullConvolution(nfiltersout, nfiltersin, kw, kh, 1, 1)

function f_fista(code, mode)
   decoder:updateOutput(code)
   local fval = fista_mse:updateOutput(decoder.output, input)
   fval = fval * 0.5

   local gradx = nil
   if mode and mode:match('dx') then
      local gradr = fista_mse:updateGradInput(decoder.output, input)
      gradr:mul(0.5)
      gradx = decoder:updateGradInput(code, gradr)
   end
   return fval, gradx
end

function g_fista(code, mode)
   local fval = l1:updateOutput(code)

   local gradx = nil
   if mode and mode:match('dx') then
      gradx = l1:updateGradInput(code)
   end
   return fval, gradx
end

function pl_fista(code, L)
   code:shrinkage(1/L)
end

params_fista = {}
params_fista.L = 0.1
params_fista.Lstep = 1.5
params_fista.maxiter = 50
params_fista.maxline = 20
params_fista.errthres = 1e-4
params_fista.doFistaUpdate = true

params_sgd = {}
params_sgd.learningRate = 2e-3
params_sgd.learningRateDecay = 1e-5

code = torch.Tensor(nfiltersout, ih - kh + 1, iw - kw + 1)

tmp = nn.Sequential()
tmp:add(encoder)
tmp:add(decoder)
x, dl_dx = tmp:getParameters()

for t = 1, maxiter do
   sample = dataset[t]
   input = sample[1]:clone()
   target = sample[2]:clone()

   function feval()
      dl_dx:zero()

      -- PSD.updateOutput
      encoder:updateOutput(input)

      code:copy(encoder.output)
      optim.FistaLS(f_fista, g_fista, pl_fista, code, params_fista)

      sgd_mse:updateOutput(encoder.output, code)

      -- PSD.updateGradInput
      sgd_mse:updateGradInput(encoder.output, code)
      encoder:updateGradInput(input, sgd_mse.gradInput)

      -- PSD.accGradParameters
      fista_mse:updateGradInput(decoder.output, input)
      fista_mse.gradInput:mul(0.5)
      decoder:accGradParameters(code, fista_mse.gradInput)
      encoder:accGradParameters(input, sgd_mse.gradInput)

      return 0, dl_dx
   end
   optim.sgd(feval, x, sgdconf)

   -- normalize
   w = decoder.weight
   for i=1,w:size(1) do
      for j=1,w:size(2) do
         local k=w:select(1,i):select(1,j)
         k:div(k:norm()+1e-12)
      end
   end


   xlua.progress(math.fmod(t, 1000), 1000)
   if math.fmod(t, 1000) == 0 then
      eweight = encoder.modules[1].weight:select(2, 1)
      dweight = decoder.weight:select(2, 1)

      dd = image.toDisplayTensor{input=dweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(nfiltersout)),
                                 symmetric=true}
      de = image.toDisplayTensor{input=eweight,
                                 padding=2,
                                 nrow=math.floor(math.sqrt(nfiltersout)),
                                 symmetric=true}

      _win1_ = gfx.image(dd, {win=_win1_, legend='d', zoom=4})
      _win2_ = gfx.image(de, {win=_win2_, legend='e', zoom=4})
   end
end
