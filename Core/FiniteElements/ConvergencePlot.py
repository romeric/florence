import numpy as np
import matplotlib.pyplot as plt

def ConvergencePlot(L2Norm,EnergyNorm,N,convergence=2):
	# if convergence==1 - it is h-convergence
	# if convergence==2 - it is p-convergence

	fig1 = plt.loglog(N,L2Norm,'-bo',label=r'$L^2\, Norm$',linewidth=2)
	fig2 = plt.loglog(N,EnergyNorm,'-r+',label=r'$Energy\, Norm$',linewidth=2)
	# plt.loglog(N,L2Norm,'-bs',N,EnergyNorm,'-ko')


	plt.xlabel(r'$n_{dof}$',{'size':'18'})
	plt.ylabel(r'$Error$',{'size':'18'})

	plt.legend()
	# plt.legend(handles=[fig2])
	
	plt.grid('on')
	# plt.savefig('/home/roman/Dropbox/Latex_Images/p_l2enorms_25.eps', format='eps', dpi=1000)
	# plt.savefig('/home/roman/Dropbox/Latex_Images/p_l2enorms_100.eps', format='eps', dpi=1000)
	# plt.savefig('/home/roman/Dropbox/Latex_Images/p_l2enorms_400.eps', format='eps', dpi=1000)
	# plt.show()



