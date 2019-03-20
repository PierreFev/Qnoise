import pyHegel.instruments

def dual_psg_set_wrap(p):
    for psg in [psg1, psg2]:
        set(psg.ampl, p)


def dual_psg_get_wrap():
    a = get(psg1.ampl)
    b = get(psg2.ampl)
    if not a == b:
        raise ValueError("The two psg in dual_psg don't have the same amplitude. They should.")
    else:
        return a


dual_psg = instruments.FunctionWrap(setfunc=dual_psg_set_wrap,
                                    getfunc=dual_psg_get_wrap,
                                    multi=('dual_psg.ampl'),
                                    getformatfunc=psg1.ampl.getformat,
                                    basedev=psg1)