from __future__ import print_function
import os, platform, sys, subprocess, imp

__all__ = ['AOTConfigure','AOTClean']



def MaterialList():
    ll_materials_mech  = [ "_NeoHookean_2_", 
                            "_MooneyRivlin_0_", 
                            "_NearlyIncompressibleMooneyRivlin_",
                            "_AnisotropicMooneyRivlin_1_"
                            ] 
    ll_materials_electro_mech  = ["_IsotropicElectroMechanics_0_", 
                            "_IsotropicElectroMechanics_3_", 
                            "_SteinmannModel_",
                            "_IsotropicElectroMechanics_101_", 
                            "_IsotropicElectroMechanics_105_", 
                            "_IsotropicElectroMechanics_106_", 
                            "_IsotropicElectroMechanics_107_",
                            "_IsotropicElectroMechanics_108_",
                            "_Piezoelectric_100_"
                        ]
    return ll_materials_mech, ll_materials_electro_mech

def execute(_cmd):
    _process = subprocess.Popen(_cmd, shell=True)
    _process.wait()



def AOTConfigure():

    ll_materials_mech, ll_materials_electro_mech = MaterialList()


    for material in ll_materials_mech:
        material_specific_assembler = "_LowLevelAssemblyDF_" + material

        header_f = material_specific_assembler + ".h"
        cython_f = material_specific_assembler + ".pyx"

        execute("cp _LowLevelAssemblyDF_.h " + header_f)
        execute("cp _LowLevelAssemblyDF_.pyx " + cython_f)

        # Read the header file
        f = open(header_f, "r")
        contents_h = f.readlines()
        f.close()

        # Modify header file
        contents_h[0] = "#ifndef " + material_specific_assembler.upper() + "_H\n"
        contents_h[1] = "#define " + material_specific_assembler.upper() + "_H\n"
        contents_h[5] = '#include "' + material + '.h"\n'
        contents_h[7] = contents_h[7].replace("_GlobalAssemblyDF_", "_GlobalAssemblyDF_"+material)

        mline = 72
        if material == "_NeoHookean_2_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu,lamb);\n"
        elif material == "_MooneyRivlin_0_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb);\n"
        elif material == "_NearlyIncompressibleMooneyRivlin_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb);\n"
        elif material == "_AnisotropicMooneyRivlin_1_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mu3,lamb);\n"
            contents_h[105] = "        mat_obj.KineticMeasures(stress, hessian, ndim, ngauss, F, &anisotropic_orientations[elem*ndim]);\n"

        contents_h[-1] = "#endif // " + material_specific_assembler.upper() + "_H"

        # Write
        f = open(header_f, 'w')
        for item in contents_h:
            f.write(item)


        # Read the header file
        f = open(cython_f, "r")
        contents_c = f.readlines()
        f.close()

        rel_line_no = -1
        # Modify cython source
        contents_c[rel_line_no+10] = 'cdef extern from "' + header_f + '" nogil:\n'
        contents_c[rel_line_no+11] = contents_c[rel_line_no+11].replace("_GlobalAssemblyDF_", "_GlobalAssemblyDF_"+material)
        contents_c[rel_line_no+102] = contents_c[rel_line_no+102].replace("_GlobalAssemblyDF_", "_GlobalAssemblyDF_"+material)
        contents_c[rel_line_no+51] = contents_c[rel_line_no+51].replace("_LowLevelAssemblyDF_", material_specific_assembler)

        mline = rel_line_no + 99
        if material == "_NeoHookean_2_":
            contents_c[mline] = "    mu, lamb = material.mu, material.lamb\n"
        elif material == "_MooneyRivlin_0_":
            contents_c[mline] = "    mu1, mu2, lamb = material.mu1, material.mu2, material.lamb\n"
        elif material == "_NearlyIncompressibleMooneyRivlin_":
            contents_c[mline] = "    mu1, mu2, lamb = material.mu1, material.mu2, material.lamb\n"
        elif material == "_AnisotropicMooneyRivlin_1_":
            contents_c[mline] = "    mu1, mu2, mu3, lamb = material.mu1, material.mu2, material.mu3, material.lamb\n"


        # Write
        f = open(cython_f, 'w')
        for item in contents_c:
            f.write(item)





    for material in ll_materials_electro_mech:
        material_specific_assembler = "_LowLevelAssemblyDPF_" + material

        header_f = material_specific_assembler + ".h"
        cython_f = material_specific_assembler + ".pyx"

        execute("cp _LowLevelAssemblyDPF_.h " + header_f)
        execute("cp _LowLevelAssemblyDPF_.pyx " + cython_f)

        # Read the header file
        f = open(header_f, "r")
        contents_h = f.readlines()
        f.close()

        # Modify header file
        contents_h[0] = "#ifndef " + material_specific_assembler.upper() + "_H\n"
        contents_h[1] = "#define " + material_specific_assembler.upper() + "_H\n"
        contents_h[5] = '#include "' + material + '.h"\n'
        contents_h[7] = contents_h[7].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_"+material)


        mline = 73
        if material == "_IsotropicElectroMechanics_0_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1);\n"
        elif material == "_IsotropicElectroMechanics_3_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1,eps_2);\n"
        elif material == "_SteinmannModel_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_3,eps_2,eps_1);\n"
        elif material == "_IsotropicElectroMechanics_101_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1);\n"
        elif material == "_IsotropicElectroMechanics_105_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_1,eps_2);\n"
        elif material == "_IsotropicElectroMechanics_106_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_1,eps_2);\n"
        elif material == "_IsotropicElectroMechanics_107_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mue,lamb,eps_1,eps_2,eps_e);\n"
        elif material == "_IsotropicElectroMechanics_108_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_2);\n"
        elif material == "_Piezoelectric_100_":
            contents_h[mline] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mu3,lamb,eps_1,eps_2,eps_3);\n"
            contents_h[118] = "        mat_obj.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx, &anisotropic_orientations[elem*ndim]);\n"

        contents_h[-1] = "#endif // " + material_specific_assembler.upper() + "_H"

        # Write
        f = open(header_f, 'w')
        for item in contents_h:
            f.write(item)


        # Read the header file
        f = open(cython_f, "r")
        contents_c = f.readlines()
        f.close()

        rel_line_no = -1
        # Modify cython source
        contents_c[rel_line_no+10] = 'cdef extern from "' + header_f + '" nogil:\n'
        contents_c[rel_line_no+11] = contents_c[rel_line_no+11].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_"+material)
        contents_c[rel_line_no+102] = contents_c[rel_line_no+102].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_"+material)
        contents_c[rel_line_no+51] = contents_c[rel_line_no+51].replace("_LowLevelAssemblyDPF_", material_specific_assembler)

        mline = rel_line_no + 99
        if material == "_IsotropicElectroMechanics_0_":
            contents_c[mline] = "    mu, lamb, eps_1 = material.mu, material.lamb, material.eps_1\n"
        elif material == "_IsotropicElectroMechanics_3_":
            contents_c[mline] = "    mu, lamb, eps_1, eps_2 = material.mu, material.lamb, material.eps_1, material.eps_2\n"
        elif material == "_SteinmannModel_":
            contents_c[mline] = "    mu, lamb, eps_3, eps_2, eps_1 = material.mu, material.lamb, material.c1, material.c2, material.eps_1\n"
        elif material == "_IsotropicElectroMechanics_101_":
            contents_c[mline] = "    mu, lamb, eps_1 = material.mu, material.lamb, material.eps_1\n"
        elif material == "_IsotropicElectroMechanics_105_":
            contents_c[mline] = "    mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2\n"
        elif material == "_IsotropicElectroMechanics_106_":
            contents_c[mline] = "    mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2\n"
        elif material == "_IsotropicElectroMechanics_107_":
            contents_c[mline] = "    mu1, mu2, mue, lamb, eps_1, eps_2, eps_e = material.mu1, " +\
                "material.mu2, material.mue, material.lamb, material.eps_1, material.eps_2, material.eps_e\n"
        elif material == "_IsotropicElectroMechanics_108_":
            contents_c[mline] = "    mu1, mu2, lamb, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_2\n"
        elif material == "_Piezoelectric_100_":
            contents_c[mline] = "    mu1, mu2, mu3, lamb, eps_1, eps_2, eps_e = material.mu1, " +\
                "material.mu2, material.mue, material.lamb, material.eps_1, material.eps_2, material.eps_e\n"

        # Write
        f = open(cython_f, 'w')
        for item in contents_c:
            f.write(item)


def AOTClean():

    ll_materials_mech, ll_materials_electro_mech = MaterialList()

    for material in ll_materials_mech:
        material_specific_assembler = "_LowLevelAssemblyDF_" + material

        header_f = material_specific_assembler + ".h"
        cython_f = material_specific_assembler + ".pyx"

        execute("rm -rf " + header_f)
        execute("rm -rf " + cython_f)


    for material in ll_materials_electro_mech:
        material_specific_assembler = "_LowLevelAssemblyDPF_" + material

        header_f = material_specific_assembler + ".h"
        cython_f = material_specific_assembler + ".pyx"

        execute("rm -rf " + header_f)
        execute("rm -rf " + cython_f)


if __name__ == "__main__":

    args = sys.argv

    _op = None

    if len(args) > 1:
        for arg in args:
            if arg == "clean" or arg == "configure":
                if _op is not None:
                    raise RuntimeError("Multiple conflicting arguments passed to setup")
                _op = arg

    if _op == "clean":
        AOTClean()
    elif _op == "configure":
        AOTConfigure()
    else:
        AOTClean()
        AOTConfigure()

