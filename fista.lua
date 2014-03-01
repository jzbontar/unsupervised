require 'cutorch'

function new_fista(f, pl, params)
   local z_prev = torch.CudaTensor()
   local z_curr = torch.CudaTensor()

   return function (z_init)
      local t_prev = 1

      z_prev:resizeAs(z_init):copy(z_init)
      z_curr:resizeAs(z_init)

      local y = z_init

      for i = 1, params.maxiter do
         local _, dfx = f(y, 'dx')
         z_curr:add(y, -1, dfx:div(params.L))
         pl(z_curr, params.L)
         local t_curr = (1 + math.sqrt(1 + 4 * t_prev^2)) / 2
         y:add(z_curr, -1, z_prev):mul((t_prev - 1) / t_curr):add(z_curr)

         t_prev = t_curr
         z_prev = z_curr
      end

      local fx, _ = f(y)
      return fx
   end
end
