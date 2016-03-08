import numpy as np
import warnings
from Florence.FunctionSpace.JacobiPolynomials import JacobiPolynomials, DiffJacobiPolynomials


def GaussQuadrature(N,a=-1,b=1):
    N0=N-1
    N1 = N0+1
    N2 = N0+2
    xu = np.linspace(-1.,1.,N1)
    
    # Legendre-Gauss-Vandermonde Matrix
    L = 1.0*np.zeros((N1,N2))
    # Derivative of Legendre-Gauss-Vandermonde Matrix
    Lp = 1.0*np.zeros(N1)

    dum = np.linspace(0,N0,N1)
    y=np.cos((2*dum+1)*np.pi/(2*N0+2))+(0.27/N1)*np.sin(np.pi*xu*N0/N2)
    # PI = np.pi
    # y=ne.evaluate("cos((2*dum+1)*PI/(2*N0+2))+(0.27/N1)*sin(PI*xu*N0/N2)")
    deps = np.finfo(np.float64).eps

    # Initial Guess
    y0 = 2.0*np.ones(N1)

    while np.max(np.abs(y-y0)) > deps:
        L[:,0] = np.ones(N1)
        L[:,1] = y
        Lp  = np.zeros(N1)

        for k in range(1,N1):
            L[:,k+1] = ((2*k+1)*L[:,k]*y - k*L[:,k-1])/(k+1)

        Lp = N2*(L[:,N0]-L[:,N1]*y)/(1-y**2)

        y0 = y
        y=y0-L[:,N1]/Lp

    z = (a*(1-y)+b*(1+y))/2.0
    w = (b-a)/((1-y**2)*Lp**2)*pow((np.float64(N2)/N1),2)


    z = np.fliplr(z.reshape(1,z.shape[0])).reshape(z.shape[0]) 
    w = np.fliplr(w.reshape(1,w.shape[0])).reshape(w.shape[0]) 

    return (z,w)





def GaussQuadratureLogarithmic(N,a=0,b=1):
    if a!=0 or b!=1:
        warnings.warn("Integral limits cannot be changed!")
    if N==20:
        zl = np.array([0.00258832795592195542833, 0.0152096623495602317207, 0.0385365503721653279598,       
            0.0721816138158739064350, 0.115460526487633150559, 0.167442856275329685718, 0.226983787260202503361, 
            0.292754960941545832992, 0.363277429857858904538, 0.436957140090768318487,  0.512122594678967336196, 
            0.587064044914409915132, 0.660073413314909413912, 0.729484083929687498871, 0.793709671987085817744, 
            0.851280892789125727222, 0.900879680854417594223, 0.941369749129091676303, 0.971822741075263193738, 
            0.991538081438711972652])

        wl = np.array([0.0431427521332080785790, 0.0753837099085893595505, 0.0930532674516630513727, 0.101456711849829754437,
            0.103201762056072069058, 0.100022549805273166533, 0.0932597993002976780837, 0.0840289528719410564971, 0.0732855891300307409628,
            0.0618503369137302899572, 0.0504166044383746776371, 0.0395513700052983853329, 0.0296940778958128448046, 0.0211563153554270976730,   
            0.0141237329389640204366, 0.00866097450433549862823, 0.00471994014620360495437, 0.00215139740396520611468, 0.000719728214653202646358,
            0.000120427676330216741693])
    elif N==10:
        zl = np.array([0.00904263096219965064, 0.0539712662225006295, 0.135311824639250775, 0.247052416287159824, 
            0.380212539609332334, 0.523792317971843201, 0.665775205516424597, 0.794190416011966217, 0.898161091219003538,
            0.968847988718633539]) 
        wl = np.array([0.120955131954570515, 0.186363542564071870, 0.195660873277759983, 0.173577142182906921,
            0.135695672995484202, 0.0936467585381105260, 0.0557877273514158741, 0.0271598108992333311, 0.00951518260284851500,
            0.00163815763359826325]) 
    elif N==5:
        zl = np.array([0.0291344721519721, 0.173977213320898, 0.411702520284902, 0.677314174582820, 0.894771361031008]) 
        wl = np.array([0.297893471782894, 0.349776226513224, 0.234488290044052, 0.0989304595166331, 0.0189115521431958])
    elif N==2:
        zl = np.array([0.11200880616697618296, 0.71853931903038444096]) 
        wl = np.array([0.60227690811873810276, 0.28146068096961555933])
    else:
        warnings.warn("Gauss points and weights are available for selected degree N: 5, 10, 20")
        wl=zl=0

    return (zl,wl)



def GaussLobattoQuadrature(C):
# def TabulatedGaussLobattoQuadrature(C):
    # THIS IS A TABULATED ROUTINE THAT FALL BACKS TO ALGORITHMIC ROUTINE FOR C>20
    zw=[]
    if C < 2:
        raise ValueError('There should be at least two Gauss-Lobatto points')
    elif C == 2:
        zw = np.array( [[-1.,  1.],
        [ 1.,  1.]] )

    elif C == 3:
        zw = np.array( [[-1.,                    0.333333333333333315],
         [ 0.,                    1.333333333333333259],
         [ 1.,                    0.333333333333333315]] )

    elif C == 4:
        zw = np.array( [[-1.   ,                 0.166666666666666657],
         [-0.447213595499957983,  0.83333333333333337 ],
         [ 0.447213595499957928,  0.83333333333333337 ],
         [ 1.                  ,  0.166666666666666657]] )

    elif C == 5:
        zw = np.array( [[-1.   ,                 0.100000000000000006],
         [-0.654653670707977198,  0.544444444444444509],
         [ 0.                  ,  0.711111111111111138],
         [ 0.654653670707977198,  0.544444444444444509],
         [ 1.                  ,  0.100000000000000006]] )

    elif C == 6:
        zw = np.array( [[-1.   ,                 0.066666666666666666],
         [-0.765055323929464626,  0.378474956297847109],
         [-0.285231516480645098,  0.554858377035486017],
         [ 0.285231516480645098,  0.554858377035486017],
         [ 0.765055323929464626,  0.378474956297847109],
         [ 1.                  ,  0.066666666666666666]] )

    elif C == 7:
        zw = np.array( [[ -1.000000000000000000e+00,   4.761904761904761640e-02],
         [ -8.302238962785669640e-01,   2.768260473615660744e-01],
         [ -4.688487934707142313e-01,   4.317453812098625554e-01],
         [ -1.232595164407830946e-32,   4.876190476190476186e-01],
         [  4.688487934707142313e-01,   4.317453812098625554e-01],
         [  8.302238962785669640e-01,   2.768260473615660744e-01],
         [  1.000000000000000000e+00,   4.761904761904761640e-02]] )

    elif C == 8:
        zw = np.array( [[-1.   ,                 0.035714285714285712],
         [-0.871740148509606683,  0.210704227143506062],
         [-0.591700181433142292,  0.341122692483504408],
         [-0.209299217902478879,  0.41245879465870372 ],
         [ 0.209299217902478879,  0.41245879465870372 ],
         [ 0.591700181433142292,  0.341122692483504408],
         [ 0.871740148509606572,  0.210704227143505951],
         [ 1.                  ,  0.035714285714285712]] )

    elif C == 9:
        zw = np.array( [[ -1.000000000000000000e+00,   2.777777777777777624e-02],
         [ -8.997579954114600653e-01,   1.654953615608056594e-01],
         [ -6.771862795107377320e-01,   2.745387125001618744e-01],
         [ -3.631174638261781551e-01,   3.464285109730463885e-01],
         [ -2.465190328815661892e-32,   3.715192743764171857e-01],
         [  3.631174638261781551e-01,   3.464285109730463885e-01],
         [  6.771862795107377320e-01,   2.745387125001618744e-01],
         [  8.997579954114600653e-01,   1.654953615608056594e-01],
         [  1.000000000000000000e+00,   2.777777777777777624e-02]] )

    elif C == 10:
        zw = np.array( [[-1.   ,                 0.02222222222222222 ],
         [-0.919533908166458858,  0.133305990851069978],
         [-0.738773865105505134,  0.224889342063126274],
         [-0.477924949810444477,  0.292042683679683612],
         [-0.165278957666387033,  0.327539761183897382],
         [ 0.165278957666387033,  0.327539761183897382],
         [ 0.477924949810444477,  0.292042683679683612],
         [ 0.738773865105505134,  0.224889342063126274],
         [ 0.919533908166458858,  0.133305990851069978],
         [ 1.                  ,  0.02222222222222222 ]] )

    elif C == 11:
        zw = np.array( [[ -1.000000000000000000e+00 ,  1.818181818181818440e-02],
         [ -9.340014304080591634e-01,   1.096122732669948530e-01],
         [ -7.844834736631444150e-01,   1.871698817803053028e-01],
         [ -5.652353269962050453e-01,   2.480481042640284295e-01],
         [ -2.957581355869394191e-01,   2.868791247790080101e-01],
         [ -2.465190328815661892e-32,   3.002175954556907111e-01],
         [  2.957581355869393636e-01,   2.868791247790081766e-01],
         [  5.652353269962050453e-01,   2.480481042640284295e-01],
         [  7.844834736631444150e-01,   1.871698817803053028e-01],
         [  9.340014304080591634e-01,   1.096122732669948530e-01],
         [  1.000000000000000000e+00,   1.818181818181818440e-02]] )

    elif C == 12:
        zw = np.array( [[-1.   ,                 0.015151515151515152],
         [-0.94489927222288228 ,  0.091684517413196151],
         [-0.819279321644006631,  0.157974705564370071],
         [-0.632876153031860733,  0.21250841776102114 ],
         [-0.399530940965348969,  0.251275603199201225],
         [-0.136552932854927561,  0.271405240910696177],
         [ 0.136552932854927561,  0.271405240910696177],
         [ 0.399530940965348913,  0.25127560319920117 ],
         [ 0.632876153031860622,  0.21250841776102114 ],
         [ 0.819279321644006631,  0.157974705564370071],
         [ 0.944899272222882169,  0.091684517413196068],
         [ 1.                  ,  0.015151515151515152]] )

    elif C == 13:
        zw = np.array( [[ -1.000000000000000000e+00,   1.282051282051282007e-02],
         [ -9.533098466421638939e-01,   7.780168674681900431e-02],
         [ -8.463475646518723305e-01,   1.349819266896082615e-01],
         [ -6.861884690817574572e-01,   1.836468652035499194e-01],
         [ -4.829098210913362332e-01,   2.207677935661100099e-01],
         [ -2.492869301062400067e-01,   2.440157903066762501e-01],
         [ -3.697785493223492838e-32,   2.519308493334467269e-01],
         [  2.492869301062399789e-01,   2.440157903066763612e-01],
         [  4.829098210913362332e-01,   2.207677935661100099e-01],
         [  6.861884690817574572e-01,   1.836468652035499194e-01],
         [  8.463475646518723305e-01,   1.349819266896082615e-01],
         [  9.533098466421638939e-01,   7.780168674681900431e-02],
         [  1.000000000000000000e+00,   1.282051282051282007e-02]] )

    elif C == 14:
        zw = np.array( [[-1.   ,                 0.01098901098901099 ],
         [-0.959935045267260922,  0.066837284497681296],
         [-0.867801053830347224,  0.116586655898711658],
         [-0.728868599091326175,  0.160021851762952028],
         [-0.550639402928647104,  0.194826149373416108],
         [-0.342724013342712852,  0.219126253009770816],
         [-0.116331868883703879,  0.231612794468457034],
         [ 0.116331868883703879,  0.231612794468457034],
         [ 0.342724013342712852,  0.219126253009770816],
         [ 0.550639402928647104,  0.194826149373416108],
         [ 0.728868599091326175,  0.160021851762952028],
         [ 0.867801053830347224,  0.116586655898711658],
         [ 0.959935045267260922,  0.066837284497681296],
         [ 1.                  ,  0.01098901098901099 ]] )

    elif C == 15:
        zw = np.array( [[ -1.000000000000000000e+00,   9.523809523809522934e-03],
         [ -9.652459265038385583e-01,   5.802989302860117604e-02],
         [ -8.850820442229763163e-01,   1.016600703257178884e-01],
         [ -7.635196899518151836e-01,   1.405116998024280306e-01],
         [ -6.062532054698457351e-01,   1.727896472536008254e-01],
         [ -4.206380547136724934e-01,   1.969872359646133164e-01],
         [ -2.153539553637942583e-01,   2.119735859268210565e-01],
         [ -3.697785493223492838e-32,   2.170481163488156284e-01],
         [  2.153539553637942583e-01,   2.119735859268210565e-01],
         [  4.206380547136724934e-01,   1.969872359646133164e-01],
         [  6.062532054698457351e-01,   1.727896472536008254e-01],
         [  7.635196899518151836e-01,   1.405116998024280306e-01],
         [  8.850820442229763163e-01,   1.016600703257178884e-01],
         [  9.652459265038385583e-01,   5.802989302860117604e-02],
         [  1.000000000000000000e+00,   9.523809523809522934e-03]] )

    elif C == 16:
        zw = np.array( [[-1.   ,                 0.008333333333333333],
         [-0.969568046270217976,  0.05085036100591997 ],
         [-0.899200533093472032,  0.089393697325930735],
         [-0.792008291861815095,  0.124255382132514094],
         [-0.652388702882493066,  0.154026980807164204],
         [-0.486059421887137633,  0.177491913391704031],
         [-0.299830468900763203,  0.193690023825203644],
         [-0.10132627352194945 ,  0.201958308178229823],
         [ 0.10132627352194945 , 0.201958308178229823],
         [ 0.299830468900763203,  0.193690023825203644],
         [ 0.486059421887137633,  0.177491913391704031],
         [ 0.652388702882493066,  0.154026980807164204],
         [ 0.792008291861815095,  0.124255382132514094],
         [ 0.899200533093472032,  0.089393697325930735],
         [ 0.969568046270217976,  0.05085036100591997 ],
         [ 1.                  ,  0.008333333333333333]] )

    elif C == 17:
        zw = np.array( [[ -1.000000000000000000e+00,   7.352941176470588133e-03],
         [ -9.731321766314182664e-01,   4.492194054325427538e-02],
         [ -9.108799959155735593e-01,   7.919827050368712096e-02],
         [ -8.156962512217703631e-01,   1.105929090070281451e-01],
         [ -6.910289806276846969e-01,   1.379877462019263867e-01],
         [ -5.413853993301015466e-01,   1.603946619976214516e-01],
         [ -3.721744335654770253e-01,   1.770042535156579055e-01],
         [ -1.895119735183173892e-01,   1.872163396776193867e-01],
         [ -4.930380657631323784e-32,   1.906618747534694347e-01],
         [  1.895119735183173892e-01,   1.872163396776193867e-01],
         [  3.721744335654770253e-01,   1.770042535156579055e-01],
         [  5.413853993301015466e-01,   1.603946619976214516e-01],
         [  6.910289806276846969e-01,   1.379877462019263867e-01],
         [  8.156962512217703631e-01,   1.105929090070281451e-01],
         [  9.108799959155735593e-01,   7.919827050368712096e-02],
         [  9.731321766314182664e-01,   4.492194054325427538e-02],
         [  1.000000000000000000e+00,   7.352941176470588133e-03]] )

    elif C == 18:
        zw = np.array( [[-1.   ,                 0.006535947712418301],
         [-0.976105557412198621,  0.039970628810913997],
         [-0.920649185347533816,  0.070637166885633762],
         [-0.835593535218090211,  0.099016271717502824],
         [-0.723679329283242634,  0.124210533132967038],
         [-0.58850483431866174 ,  0.145411961573802206],
         [-0.434415036912123964,  0.161939517237602498],
         [-0.266362652878280981,  0.173262109489456279],
         [-0.089749093484652112,  0.179015863439702994],
         [ 0.089749093484652112,  0.179015863439702994],
         [ 0.266362652878280981,  0.173262109489456279],
         [ 0.434415036912123964,  0.161939517237602498],
         [ 0.58850483431866174 ,  0.145411961573802206],
         [ 0.723679329283242634,  0.124210533132967038],
         [ 0.835593535218090211,  0.099016271717502824],
         [ 0.920649185347533816,  0.070637166885633762],
         [ 0.976105557412198621,  0.039970628810913997],
         [ 1.                  ,  0.006535947712418301]] )

    elif C == 19:
        zw = np.array( [[ -1.000000000000000000e+00,   5.847953216374268681e-03],
         [ -9.786117662220801261e-01,   3.579336518617644985e-02],
         [ -9.289015281525861978e-01,   6.338189176262973290e-02],
         [ -8.524605777966460796e-01,   8.913175709920712064e-02],
         [ -7.514942025526130109e-01,   1.123153414773050557e-01],
         [ -6.289081372652205459e-01,   1.322672804487509124e-01],
         [ -4.882292856807134984e-01,   1.484139425959388470e-01],
         [ -3.335048478244986292e-01,   1.602909240440612837e-01],
         [ -1.691860234092815718e-01,   1.675565845271427823e-01],
         [ -4.930380657631323784e-32,   1.700019192848272187e-01],
         [  1.691860234092815718e-01,   1.675565845271427823e-01],
         [  3.335048478244986292e-01,   1.602909240440612837e-01],
         [  4.882292856807134984e-01,   1.484139425959388470e-01],
         [  6.289081372652205459e-01,   1.322672804487509124e-01],
         [  7.514942025526130109e-01,   1.123153414773050557e-01],
         [  8.524605777966460796e-01,   8.913175709920712064e-02],
         [  9.289015281525861978e-01,   6.338189176262973290e-02],
         [  9.786117662220801261e-01,   3.579336518617644985e-02],
         [  1.000000000000000000e+00,   5.847953216374268681e-03]] )

    elif C == 20:
        zw = np.array( [[-1.   ,                 0.005263157894736842],
         [-0.980743704893914159,  0.032237123188488953],
         [-0.935934498812665439,  0.057181802127566697],
         [-0.866877978089950152,  0.080631763996119529],
         [-0.77536826095205591 ,  0.101991499699450802],
         [-0.663776402290311318,  0.120709227628674726],
         [-0.534992864031886284,  0.136300482358724134],
         [-0.392353183713909315,  0.148361554070916835],
         [-0.239551705922986496,  0.156580102647475461],
         [-0.080545937238821835,  0.160743286387845769],
         [ 0.080545937238821849,  0.160743286387845852],
         [ 0.239551705922986496, 0.156580102647475461],
         [ 0.392353183713909315,  0.148361554070916835],
         [ 0.534992864031886284,  0.136300482358724134],
         [ 0.663776402290311318,  0.120709227628674726],
         [ 0.77536826095205591 ,  0.101991499699450802],
         [ 0.866877978089950152,  0.080631763996119529],
         [ 0.935934498812665439,  0.057181802127566697],
         [ 0.980743704893914159,  0.032237123188488953],
         [ 1.                  ,  0.005263157894736842]] )

    elif C>20:
        # FALL BACK TO ALGORITHMIC PROCEDURE
        # Build Jacobi polynomial with coefficients a=b=1;
        a=1;    b=a;
        # Redefine C: at least there are two Gauss-Lobatto points always
        if C<=1:
            raise ValueError("There should be at least two Gauss-Lobatto points")
        if C>=1:
            C=C-2
        # Initial Guess - Chebyshev-Gauss-Lobatto points
        x=-np.cos(np.linspace(0,C+1,C+2)/(C+1)*np.pi) 
        # Allocate space for points and weights
        z = np.zeros((x.shape[0],1)); w = np.copy(z)
        for k in range(0,x.shape[0]):
            x0=x[k]
            dell = 2 
            while np.abs(dell) > np.finfo(float).eps:
                # Polynomial Deflation: Exclude already determined roots
                s = np.sum(1.0/(x0-z[0:k]))
                # Compute Jacobi polynomial p(a,b)
                p = JacobiPolynomials(C,x0,a,b)
                # Compute derivative of Jacobi polynomial p(a,b)
                dp = DiffJacobiPolynomials(C,x0,a,b,1)
                # Gauss-Lobatto points are roots of (1-x^2)*dp, hence 
                nom = (1.0-x0**2)*p[C]    
                dnom = -2.0*x0*p[C]+(1-x0**2)*dp[C]
                dell = - nom/(dnom-nom*s)
                x1 = x0+dell
                x0=x1
            z[k] = x1
            # Compute weights 
            p1 = JacobiPolynomials(C+1,x1,0,0)
            w[k] = 2.0/(C+1)/(C+2)/p1[C+1]**2

        zw = np.concatenate((z,w),axis=1)

    return zw[:,0].reshape(zw.shape[0],1),zw[:,1].reshape(zw.shape[0],1)




# def GaussLobattoQuadrature(C):
#   # Build Jacobi polynomial with coefficients a=b=1;
#   a=1;    b=a;
#   # Redefine C: at least there are two Gauss-Lobatto points always
#   if C<=1:
#       warnings.warn("There should be at least two Gauss-Lobatto points")
#   if C>=1:
#       C=C-2
#   # Initial Guess - Chebyshev-Gauss-Lobatto points
#   x=-np.cos(np.linspace(0,C+1,C+2)/(C+1)*np.pi) 
#   # Allocate space for points and weights
#   z = np.zeros((x.shape[0],1)); w = np.copy(z)
#   for k in range(0,x.shape[0]):
#       x0=x[k]
#       dell = 2 
#       while np.abs(dell) > np.finfo(float).eps:
#           # Polynomial Deflation: Exclude already determined roots
#           s = np.sum(1.0/(x0-z[0:k]))
#           # Compute Jacobi polynomial p(a,b)
#           p = JacobiPolynomials(C,x0,a,b)
#           # Compute derivative of Jacobi polynomial p(a,b)
#           dp = DiffJacobiPolynomials(C,x0,a,b,1)
#           # Gauss-Lobatto points are roots of (1-x^2)*dp, hence 
#           nom = (1.0-x0**2)*p[C]    
#           dnom = -2.0*x0*p[C]+(1-x0**2)*dp[C]
#           dell = - nom/(dnom-nom*s)
#           x1 = x0+dell
#           x0=x1
#       z[k] = x1
#       # Compute weights 
#       p1 = JacobiPolynomials(C+1,x1,0,0)
#       w[k] = 2.0/(C+1)/(C+2)/p1[C+1]**2

#   return (z,w)
