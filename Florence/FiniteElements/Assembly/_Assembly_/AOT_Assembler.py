from __future__ import print_function
import os, platform, sys, subprocess, imp

__all__ = ['AOTConfigure','AOTClean']



def MaterialList():
    ll_materials_mech          = [  "_LinearElastic_",
                                    "_NeoHookean_",
                                    "_MooneyRivlin_",
                                    "_NearlyIncompressibleMooneyRivlin_",
                                    "_AnisotropicMooneyRivlin_1_"
                                    ]
    ll_materials_electro_mech  = [  "_IsotropicElectroMechanics_0_",
                                    "_IsotropicElectroMechanics_3_",
                                    "_SteinmannModel_",
                                    "_IsotropicElectroMechanics_101_",
                                    "_IsotropicElectroMechanics_105_",
                                    "_IsotropicElectroMechanics_106_",
                                    "_IsotropicElectroMechanics_107_",
                                    "_IsotropicElectroMechanics_108_",
                                    "_Piezoelectric_100_"
                                    ]

    # ll_materials_mech  = []
    # ll_materials_electro_mech  = ["_IsotropicElectroMechanics_108_"]
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

        for counter, line in enumerate(contents_h):
            if "_GlobalAssemblyDF_" in line:
                contents_h[counter] = contents_h[counter].replace("_GlobalAssemblyDF_", "_GlobalAssemblyDF_"+material)

            if "auto mat_obj" in line:
                if material == "_NeoHookean_" or material == "_LinearElastic_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu,lamb);\n"
                elif material == "_MooneyRivlin_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb);\n"
                elif material == "_NearlyIncompressibleMooneyRivlin_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb);\n"
                elif material == "_AnisotropicMooneyRivlin_1_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mu3,lamb);\n"

        if material == "_AnisotropicMooneyRivlin_1_":
            for counter, line in enumerate(contents_h):
                if "mat_obj.KineticMeasures" in line:
                    contents_h[counter] = "        mat_obj.KineticMeasures(stress, hessian, ndim, ngauss, F, &anisotropic_orientations[elem*ndim]);\n"

        # Turn off geometric stiffness
        if material == "_LinearElastic_":
            for counter, line in enumerate(contents_h):
                if "std::fill" in line and "geometric_stiffness" in line:
                    contents_h[counter] = ""
                # DANGEROUS - this is relative to current line - we delete 13 lines down
                if "_GeometricStiffnessFiller_" in line:
                    contents_h[counter:counter+13] = ""


        contents_h[-1] = "#endif // " + material_specific_assembler.upper() + "_H"

        # Write
        f = open(header_f, 'w')
        for item in contents_h:
            f.write(item)


        # Read the cython wrapper file
        f = open(cython_f, "r")
        contents_c = f.readlines()
        f.close()


        for counter, line in enumerate(contents_c):
            if "_LowLevelAssemblyDF_" in line:
                contents_c[counter] = contents_c[counter].replace("_LowLevelAssemblyDF_", "_LowLevelAssemblyDF_" + material)
            if "_GlobalAssemblyDF_" in line:
                contents_c[counter] = contents_c[counter].replace("_GlobalAssemblyDF_", "_GlobalAssemblyDF_" + material)

            if "mu1, mu2, lamb =" in line:
                if material == "_NeoHookean_" or material == "_LinearElastic_":
                    contents_c[counter] = "    mu, lamb = material.mu, material.lamb\n"
                elif material == "_MooneyRivlin_":
                    contents_c[counter] = "    mu1, mu2, lamb = material.mu1, material.mu2, material.lamb\n"
                elif material == "_NearlyIncompressibleMooneyRivlin_":
                    contents_c[counter] = "    mu1, mu2, lamb = material.mu1, material.mu2, material.lamb\n"
                elif material == "_AnisotropicMooneyRivlin_1_":
                    contents_c[counter] = "    mu1, mu2, mu3, lamb = material.mu1, material.mu2, material.mu3, material.lamb\n"

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
        # contents_h[7] = contents_h[7].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_"+material)

        for counter, line in enumerate(contents_h):
            if "_GlobalAssemblyDPF_" in line:
                contents_h[counter] = contents_h[counter].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_"+material)

            if "auto mat_obj" in line:
                if material == "_IsotropicElectroMechanics_0_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1);\n"
                elif material == "_IsotropicElectroMechanics_3_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1,eps_2);\n"
                elif material == "_SteinmannModel_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_3,eps_2,eps_1);\n"
                elif material == "_IsotropicElectroMechanics_101_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu,lamb,eps_1);\n"
                elif material == "_IsotropicElectroMechanics_105_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_1,eps_2);\n"
                elif material == "_IsotropicElectroMechanics_106_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_1,eps_2);\n"
                elif material == "_IsotropicElectroMechanics_107_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mue,lamb,eps_1,eps_2,eps_e);\n"
                elif material == "_IsotropicElectroMechanics_108_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,lamb,eps_2);\n"
                elif material == "_Piezoelectric_100_":
                    contents_h[counter] = "    auto mat_obj = " + material + "<Real>(mu1,mu2,mu3,lamb,eps_1,eps_2,eps_3);\n"


        if material == "_Piezoelectric_100_":
            for counter, line in enumerate(contents_h):
                if "mat_obj.KineticMeasures" in line:
                    contents_h[counter] = "        mat_obj.KineticMeasures(D, stress, hessian, ndim, ngauss, F, ElectricFieldx, &anisotropic_orientations[elem*ndim]);\n"

        contents_h[-1] = "#endif // " + material_specific_assembler.upper() + "_H"

        # Write
        f = open(header_f, 'w')
        for item in contents_h:
            f.write(item)


        # Read the header file
        f = open(cython_f, "r")
        contents_c = f.readlines()
        f.close()

        for counter, line in enumerate(contents_c):
            if "_LowLevelAssemblyDPF_" in line:
                contents_c[counter] = contents_c[counter].replace("_LowLevelAssemblyDPF_", "_LowLevelAssemblyDPF_" + material)
            if "_GlobalAssemblyDPF_" in line:
                contents_c[counter] = contents_c[counter].replace("_GlobalAssemblyDPF_", "_GlobalAssemblyDPF_" + material)

            if "mu, lamb, eps_1 =" in line:
                if material == "_IsotropicElectroMechanics_0_":
                    contents_c[counter] = "    mu, lamb, eps_1 = material.mu, material.lamb, material.eps_1\n"
                elif material == "_IsotropicElectroMechanics_3_":
                    contents_c[counter] = "    mu, lamb, eps_1, eps_2 = material.mu, material.lamb, material.eps_1, material.eps_2\n"
                elif material == "_SteinmannModel_":
                    contents_c[counter] = "    mu, lamb, eps_3, eps_2, eps_1 = material.mu, material.lamb, material.c1, material.c2, material.eps_1\n"
                elif material == "_IsotropicElectroMechanics_101_":
                    contents_c[counter] = "    mu, lamb, eps_1 = material.mu, material.lamb, material.eps_1\n"
                elif material == "_IsotropicElectroMechanics_105_":
                    contents_c[counter] = "    mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2\n"
                elif material == "_IsotropicElectroMechanics_106_":
                    contents_c[counter] = "    mu1, mu2, lamb, eps_1, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_1, material.eps_2\n"
                elif material == "_IsotropicElectroMechanics_107_":
                    contents_c[counter] = "    mu1, mu2, mue, lamb, eps_1, eps_2, eps_e = material.mu1, " +\
                        "material.mu2, material.mue, material.lamb, material.eps_1, material.eps_2, material.eps_e\n"
                elif material == "_IsotropicElectroMechanics_108_":
                    contents_c[counter] = "    mu1, mu2, lamb, eps_2 = material.mu1, material.mu2, material.lamb, material.eps_2\n"
                elif material == "_Piezoelectric_100_":
                    contents_c[counter] = "    mu1, mu2, mu3, lamb, eps_1, eps_2, eps_3 = material.mu1, " +\
                        "material.mu2, material.mu3, material.lamb, material.eps_1, material.eps_2, material.eps_3\n"

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

