function fista(f, g, pl, z_init, params)
   local y = {}
   local z = {}
   local t = {}

   z[0] = z_init
   y[1] = z[0]
   t[1] = 1
   for i = 1, params.maxiter do
      local fx, dfx = f(y[i], 'dx')
      z[i] = torch.CudaTensor():resizeAs(z_init)
      y[i + 1] = torch.CudaTensor():resizeAs(z_init)

      z[i] = z[i]:add(y[i], -1, dfx:div(params.L))
      pl(z[i], params.L)
      t[i + 1] = (1 + math.sqrt(1 + 4 * t[i]^2)) / 2
      y[i + 1]:add(z[i], -1, z[i - 1]):mul((t[i] - 1) / t[i + 1]):add(z[i])
   end
   z_init:copy(y[#y])
   local fx = f(z_init)
   return fx
end
