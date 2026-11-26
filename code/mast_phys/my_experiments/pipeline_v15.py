"""Release v15: valid laser pulses, stable equilibrium, CUDA physical readout."""
import numpy as np
import torch
from my_experiments.mast_equilibrium_v13 import ThomsonEquilibriumV13
from my_experiments.mast_inputs_v15 import prepare_kinetic
from my_experiments.temporal_guard_v13 import guard_current


def infer_fields(eq,batch,steps=600):
    """batch entries contain row, times, kinetic constraints, and currents."""
    kin=np.concatenate([b['kin'] for b in batch]);curr=np.concatenate([b['curr'] for b in batch])
    thermal=tuple(np.concatenate([b['thermal'][j] for b in batch]) for j in range(3))
    params,history,loss=eq.optimize_prepared(kin,curr,steps,thermal);seeds=[];off=0
    for b in batch:
        t=b['times'];n=len(t);p=params[off:off+n];off+=n
        width=max(12.,2*np.median(np.diff(t)) if n>1 else 12.)
        for j in range(n):seeds.append(np.median(p[abs(t-t[j])<=width],axis=0))
    warm,wh,wl=eq.optimize_prepared(kin,curr,max(100,steps//2),thermal,np.asarray(seeds))
    choose=wl<loss;params[choose]=warm[choose];loss[choose]=wl[choose]
    output=[];off=0
    for b in batch:
        t=b['times'];cu=b['curr'];n=len(t);state=params[off:off+n];errors=loss[off:off+n];off+=n
        phi=eq.reconstruct(state,cu)
        with torch.no_grad():
            w=eq.source(eq.t(state),fine=True);response=eq.grid_response(w)
            fixed,guard=guard_current(t,w.cpu().numpy(),response.cpu().numpy(),cu)
            changed=np.asarray([g['frame'] for g in guard],int)
            if len(changed):
                plasma=eq.grid_response(eq.t(fixed[changed])*eq.t(cu[changed,-1]*1000)[:,None]).cpu().numpy()
                phi[changed]=plasma+np.einsum('tc,czr->tzr',cu[changed]*1000,eq.basis,optimize=True)
        output.append((phi,{'parameters':state.tolist(),'mean_loss':float(errors.mean()),'guards':guard}))
    return output,{'initial':history,'restart':wh,'restart_selected_frames':int(choose.sum())}
