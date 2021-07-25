#include <CobraTypesDefine.h>
#include <spdlog/spdlog.h>
#define CPT_NN_TRIANGLE 0
namespace cobra
{

	class  OptimizedSdf
	{
	public:
		static void ParellComputeUSdf(
			const std::vector<Vector3> &point, const std::vector<int> &face,
			const Real sample, const Vector3& minp,
			const unsigned int udim, const unsigned int vdim, const unsigned int wdim,
			bool flag, unsigned int smoothTimes,
#if CPT_NN_TRIANGLE
			unsigned int*& neastTriangle,
#endif
			Real*& sdfValue);



	protected:
		//FIM解方程,参考论文 A FAST ITERATIVE METHOD FOR EIKONAL EQUATIONS 实现
		static  void SolveOnIsotropGrid(
			unsigned ni, unsigned nj, unsigned nk,
			Real lbx, Real lby, Real lbz,
			Real h, Real* sol, 
#if CPT_NN_TRIANGLE
			const Vector3*pointD,const int* faced, unsigned int*& neastTriangle,
#endif
			Real* speed = NULL,
			Real max_float = std::numeric_limits< Real >::max());
	protected:

	protected:
		static std::shared_ptr<spdlog::logger> m_logger;
	};
}
