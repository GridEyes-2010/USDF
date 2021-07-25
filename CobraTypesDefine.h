#pragma once
#pragma warning(push)
#pragma warning(disable: 4127)
#include <Eigen/Eigen>
#pragma warning(pop)
namespace cobra
{

	using Real = float;
	using IndexType = int;

	template<int DIM>
	using Vector = Eigen::Matrix<Real, DIM, 1>;

	template<int DIM>
	using Vectord = Eigen::Matrix<double, DIM, 1>;

	template<int DIM>
	using Det = Eigen::Matrix<Real, DIM, DIM>;

	template<int rol,int col>
	using Matrix = Eigen::Matrix<Real, rol, col>;
	using MatrixN = Matrix<Eigen::Dynamic, Eigen::Dynamic>;
	

	using SparseMatrix = Eigen::SparseMatrix<Real>;


	using Vector2 = Vector<2>;
	using Vector3 = Vector<3>;
	using Vector4 = Vector<4>;
	using VectorN = Eigen::Matrix<Real, Eigen::Dynamic, 1>;

	using Polygon2 = std::vector<Vector2>;

	using RGBI = Eigen::Matrix<unsigned short,3,1>;
	using RGB = Eigen::Matrix<Real,3,1>;
	using RGBA = Eigen::Matrix<Real, 4, 1>;

	using TexCoord = Vector<3>;
	using Normal = Vector<3>;

	using Vector2d = Vectord<2>;
	using Vector3d = Vectord<3>;
	using Vector4d = Vectord<4>;

	using Det3 = Det<3>;
	using Det4 = Det<4>;
#ifdef _DEBUG
#define CobraAssert(x) assert(x)
#else
#define CobraAssert(x) 
#endif


	struct RGBUtil {
		//------- conversion between a RGB(A) color and an integer ----------

		/// \brief Gets red part of RGB. [0, 255]
		static inline int red(int color) { return ((color >> 16) & 0xff); }

		/// \brief Gets green part of RGB. [0, 255]
		static inline int green(int color) { return ((color >> 8) & 0xff); }

		/// \brief Gets blue part of RGB. [0, 255]
		static inline int blue(int color) { return (color & 0xff); }

		/// \brief Gets alpha part of RGBA. [0, 255]
		static inline int alpha(int color) { return color >> 24; }

		/// \brief Encodes an RGB (each component in the range [0, 255]) color in an integer value.
		static inline int rgb(int r, int g, int b) {
			return (0xffu << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
		}

		/// \brief Encodes an RGBA (each component in the range [0, 255]) color in an integer value.
		static inline int rgba(int r, int g, int b, int a) {
			return ((a & 0xff) << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
		}

		/// \brief Converts R,G,B to gray [0, 255]
		static inline int gray(int r, int g, int b) {
			return (r * 11 + g * 16 + b * 5) >>5;
		}

		/// \brief Decodes an integer value as RGB color (each component in the range [0, 255]).
		static inline void encode(int value, int &r, int &g, int &b) {
			r = ((value >> 16) & 0xff);
			g = ((value >> 8) & 0xff);
			b = (value & 0xff);
		}

		/// \brief Decodes an integer value as RGBA color (each component in the range [0, 255]).
		static inline void encode(int value, int &r, int &g, int &b, int &a) {
			r = ((value >> 16) & 0xff);
			g = ((value >> 8) & 0xff);
			b = (value & 0xff);
			a = (value >> 24);
		}
	};

	template<class T>
	inline const T CobraEps() {
		return static_cast<T>(1E-4F);
	}


	template<class T>
	inline const T CobraPI() {
		return static_cast<T>(3.14159265358979323846);
	}

	template<class T>
	inline const T CobraCommomEps() {
		return static_cast<T>(FLT_EPSILON);
	}

	template<typename T>
	inline void hash_combine(uint64_t &seed, T const& value) {
		std::hash<T> hasher;
		uint64_t a = (hasher(value) ^ seed) * 0x9ddfea08eb382d69ULL;
		a ^= (a >> 47);
		uint64_t b = (seed ^ a) * 0x9ddfea08eb382d69ULL;
		b ^= (b >> 47);
		seed = b * 0x9ddfea08eb382d69ULL;
	}
	template<typename Iterator>
	inline uint64_t hash(Iterator first, Iterator last) {
		uint64_t seed = 0;
		for (; first != last; ++first) {
			hash_combine(seed, *first);
		}
		return seed;
	}

	class CmpVec {
	public:
		explicit CmpVec(float _eps = FLT_MIN) : eps_(_eps) {}

		bool operator()(const Vector3 &v0, const Vector3 &v1) const {
			if (fabs(v0[0] - v1[0]) <= eps_) {
				if (fabs(v0[1] - v1[1]) <= eps_) {
					return (v0[2] < v1[2] - eps_);
				}
				else {
					return (v0[1] < v1[1] - eps_);
				}
			}
			else {
				return (v0[0] < v1[0] - eps_);
			}
		}

	private:
		float eps_;
	};


	struct Vec3Hash
	{
		void HashCombine(size_t& seed, const Real& v)const
		{
			std::hash<Real> hasher;
			seed ^= hasher(v);
		}
		template<class Point>
		size_t operator()(const Point& v) const
		{
			size_t seed = 0;
			HashCombine(seed, v[0]);
			HashCombine(seed, v[1]);
			HashCombine(seed, v[2]);
			return seed;
		}
	};


	struct Vec3Equal
	{
		Vec3Equal(Real eps = 0) :m_eps(eps) {}
		template<class Point>
		bool operator()(const Point& a, const Point& b) const
		{
			return (a[0]-b[0])*(a[0] - b[0])+ (a[1] - b[1])*(a[1] - b[1])+ (a[2] - b[2])*(a[2] - b[2]) <= m_eps;
		}
		Real m_eps;
	};

}
