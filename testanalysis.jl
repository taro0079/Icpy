using DataFrames
using CSV
using Plots
using Dierckx

df = CSV.read("./20201008A/data/test", DataFrame);
ind = nonunique(df,2)
delete!(df, ind)
current = df[:, 2];
voltage = df[:, 3];
itp = Spline1D(current, voltage)
xx = range(0, stop=1, length=2000)

gr()
plot(xx, itp(xx));
