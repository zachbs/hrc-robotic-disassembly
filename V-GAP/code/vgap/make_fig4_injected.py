import pandas as pd,glob,numpy as np,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm,glob as _g
for _f in _g.glob('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'): fm.fontManager.addfont(_f)
plt.rcParams.update({'font.family':'serif','font.serif':['TeX Gyre Termes'],'mathtext.fontset':'custom','mathtext.rm':'TeX Gyre Termes','mathtext.it':'TeX Gyre Termes:italic','mathtext.bf':'TeX Gyre Termes:bold'})
df=pd.concat([pd.read_csv(f) for f in glob.glob('/mnt/project-files/data/gate_eval_all/*/injected.csv')])
w=df[df.grade=='wrong']
objs=[('gearbox','Gearbox'),('l_block','L-block'),('t_block','T-block'),('hemisphere','Hemisphere')]
a,b=[],[]
for o,_ in objs:
    g=w[w.object==o]; thr=0.6 if o=='hemisphere' else 0.7
    acc=g.gate=='Valid'; a.append(100*acc.mean()); b.append(100*(acc&(g.within_3mm>=thr)).mean())
fig,ax=plt.subplots(figsize=(3.5,2.75),dpi=300)
x=np.arange(4); wd=0.38
ax.bar(x-wd/2,a,wd,color='#EB6834',label='2D consistency score ($C$) only')
ax.bar(x+wd/2,b,wd,color='#2A79D6',label='2D consistency score ($C$) +\n3D fit criterion ($F_{3D}$)')
for i in range(4):
    ax.text(x[i]-wd/2,a[i]+0.8,f'{a[i]:.1f}',ha='center',fontsize=10)
    ax.text(x[i]+wd/2,b[i]+0.8,f'{b[i]:.1f}',ha='center',fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([l for _,l in objs],fontsize=10.5)
ax.set_ylabel('Wrong poses accepted (%)',fontsize=11.5); ax.tick_params(axis='y',labelsize=10.5)
ax.set_ylim(0,54); ax.set_yticks(range(0,41,10)); ax.spines[['top','right']].set_visible(False); ax.grid(False)
ax.legend(fontsize=10,frameon=False,loc='upper left',handlelength=1.2,borderaxespad=0.1)
fig.tight_layout(pad=0.3); fig.savefig(sys.argv[1],dpi=300)
