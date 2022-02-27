#include <vector>
#include <adios2.h>
int main(int argc, char *argv[])
{
    std::vector<double> myData = {
        0.0001, 1.0001, 2.0001, 3.0001, 4.0001, 5.0001, 6.0001, 7.0001, 8.0001, 9.0001,
        1.0001, 2.0001, 3.0001, 4.0001, 5.0001, 6.0001, 7.0001, 8.0001, 9.0001, 8.0001,
        2.0001, 3.0001, 4.0001, 5.0001, 6.0001, 7.0001, 8.0001, 9.0001, 8.0001, 7.0001,
        3.0001, 4.0001, 5.0001, 6.0001, 7.0001, 8.0001, 9.0001, 8.0001, 7.0001, 6.0001,
        4.0001, 5.0001, 6.0001, 7.0001, 8.0001, 9.0001, 8.0001, 7.0001, 6.0001, 5.0001,
        5.0001, 6.0001, 7.0001, 8.0001, 9.0001, 8.0001, 7.0001, 6.0001, 5.0001, 4.0001,
        6.0001, 7.0001, 8.0001, 9.0001, 8.0001, 7.0001, 6.0001, 5.0001, 4.0001, 3.0001,
        7.0001, 8.0001, 9.0001, 8.0001, 7.0001, 6.0001, 5.0001, 4.0001, 3.0001, 2.0001,
        8.0001, 9.0001, 8.0001, 7.0001, 6.0001, 5.0001, 4.0001, 3.0001, 2.0001, 1.0001,
        9.0001, 8.0001, 7.0001, 6.0001, 5.0001, 4.0001, 3.0001, 2.0001, 1.0001, 0.0001,
    };
    adios2::ADIOS adios;
    auto io = adios.DeclareIO("TestIO");
    auto varDouble = io.DefineVariable<double>("varDouble", {10,10}, {0,0}, {10,10}, adios2::ConstantDims);

    // add operator
    varDouble.AddOperation("mgardplus",{{"accuracy","0.01"}});
    // end add operator

    auto engine = io.Open("hello.bp", adios2::Mode::Write);
    engine.Put<double>(varDouble, myData.data());
    engine.Close();
    return 0;
}