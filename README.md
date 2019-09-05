plot([x->del(Vec((x,0.0)))/1e3, x->4pi*pot(Vec((x,0.0))), x->max(-10, -1/abs(x))], -1, 1, color=["delta", "potential", "1/r"])
