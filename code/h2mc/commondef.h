#pragma once
#include <vector>
#include <random>
#include <cmath>
#include "./Eigen/Dense"


template <typename T, int N, int _Options = Eigen::AutoAlign>
using TVector = Eigen::Matrix<T, N, 1, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TVector1 = TVector<T, 1, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TVector2 = TVector<T, 2, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TVector3 = TVector<T, 3, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TVector4 = TVector<T, 4, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TMatrix3x3 = Eigen::Matrix<T, 3, 3, _Options>;
template <typename T, int _Options = Eigen::AutoAlign>
using TMatrix4x4 = Eigen::Matrix<T, 4, 4, _Options>;

using Float = float;
using AlignedStdVector = std::vector<Float>;

using Vector1 = TVector1<Float>;
using Vector2 = TVector2<Float>;
using Vector3 = TVector3<Float>;
using Vector4 = TVector4<Float>;
using Matrix3x3 = TMatrix3x3<Float>;
using Matrix4x4 = TMatrix4x4<Float>;
using Vector2i = TVector2<int>;

using Vector = Eigen::Matrix<Float, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>;

using RNG = std::mt19937;

const Float PI = Float(3.141592653589793238462643383279502884197169399375105820974944592307816406);

