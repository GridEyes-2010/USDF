#include <USDF.h>
#include <cstdint>
#include <algorithm>
#include <fcpw/fcpw.h>
#include <ppl.h>
#include <ppltasks.h>
namespace cobra
{
	namespace {
		template<class T>
		inline T Clamp(T a, T b, T c) {
			if (a < b)return b;
			if (a > c)return c;
			return a;
		}
	}
	std::shared_ptr<spdlog::logger> OptimizedSdf::m_logger = spdlog::stdout_color_mt("OptimizedSdf");

	void OptimizedSdf::ParellComputeUSdf(
		const std::vector<Vector3> &point, const std::vector<int> &face,
		const Real sample, const Vector3& minp,
		const unsigned int udim, const unsigned int vdim, const unsigned int wdim,
		bool flag, unsigned int smoothTimes,
#if CPT_NN_TRIANGLE
		unsigned int*& neastTriangle,
#endif
		Real*& sdfValue)
	{
		clock_t st = clock();
		unsigned int nTriangleSize = static_cast<unsigned int>(face.size() / 3);
		unsigned int nPointSize = static_cast<unsigned int>(point.size());
		std::shared_ptr<fcpw::Scene<3>> m_pTree = nullptr;
		const int* faced = face.data();
		const Vector3* pointd = point.data();
		const unsigned int uzero = 0;
		const Real invSample = 1 / sample;
		const unsigned int exactBand = smoothTimes;
		const unsigned int vudim = vdim * udim;
		unsigned int totalSize = udim * vdim*wdim;
		unsigned char* mark=nullptr;
		Vector3* markpts=nullptr;

		concurrency::task<void> buildTree([&]()
		{
			m_pTree = std::make_shared<fcpw::Scene<3>>();
			m_pTree->setObjectTypes({ {fcpw::PrimitiveType::Triangle} });
			m_pTree->setObjectVertexCount((int)nPointSize, 0);
			m_pTree->setObjectTriangleCount((int)nTriangleSize, 0);

			for (unsigned i = 0; i < nPointSize; ++i) {
				m_pTree->setObjectVertex(point[i], i, 0);
			}
			for (unsigned i = 0; i < nTriangleSize; ++i) {
				m_pTree->setObjectTriangle(&faced[i + i + i], i, 0);
			}
			m_pTree->build(fcpw::AggregateType::Bvh_SurfaceArea, true);
		});

		concurrency::task<void> applyforResources([&]()
		{
			sdfValue = nullptr;
			sdfValue = (Real*)malloc(sizeof(Real)*totalSize);
			if (nullptr == sdfValue) {
				throw std::runtime_error("malloc sdfValue error!");
			}
			else {
				std::fill_n(&sdfValue[0], totalSize, std::numeric_limits<Real>::max());
			}
#if CPT_NN_TRIANGLE
			neastTriangle = nullptr;
			neastTriangle = (unsigned int*)malloc(sizeof(unsigned int)*totalSize);
			if (nullptr == sdfValue) {
				throw std::runtime_error("malloc neastTriangle error!");
			}
			else {
				std::fill_n(&sdfValue[0], totalSize, nPointSize);
			}
#endif			
			{
				mark = (unsigned char*)malloc(sizeof(unsigned char)*totalSize);
				std::fill_n(&mark[0], totalSize, 0);
				markpts = (Vector3*)malloc(sizeof(Vector3)*totalSize);
				Vector3 vp;
				Real wsample = 0;
				for (unsigned w = uzero; w < wdim; ++w) {
					vp[2] = minp[2] + wsample;
					const unsigned wvu = w * vudim;
					Real vsample = 0;
					for (unsigned v = uzero; v < vdim; ++v) {
						vp[1] = minp[1] + vsample;
						const unsigned vd = wvu + v * udim;
						Real usample = 0;
						for (unsigned u = uzero; u < udim; ++u) {
							vp[0] = minp[0] + usample;
							markpts[vd + u] = vp;
							usample += sample;
						}
						vsample += sample;
					}
					wsample += sample;
				}
			}
		});
		buildTree.wait();
		applyforResources.wait();
		m_logger->info("applyforResources Time:{}ms", clock() - st);
		st = clock();
		concurrency::parallel_for(uzero, nTriangleSize, [&](unsigned int nTri) {
			unsigned idt = nTri + nTri + nTri;
			int tp0 = faced[idt];
			int tp1 = faced[idt + 1];
			int tp2 = faced[idt + 2];

			Vector3 m_min = pointd[tp0];
			Vector3 m_max = pointd[tp0];
			m_min = m_min.cwiseMin(pointd[tp1]);
			m_min = m_min.cwiseMin(pointd[tp2]);
			m_max = m_max.cwiseMax(pointd[tp1]);
			m_max = m_max.cwiseMax(pointd[tp2]);
			m_min -= minp;
			m_max -= minp;

			m_min *= invSample;
			m_max *= invSample;

			unsigned us = Clamp((unsigned int)m_min[0] - exactBand, uzero, udim - 1);
			unsigned ue = Clamp((unsigned int)m_max[0] + exactBand + 1, uzero, udim - 1);
			unsigned vs = Clamp((unsigned int)m_min[1] - exactBand, uzero, vdim - 1);
			unsigned ve = Clamp((unsigned int)m_max[1] + exactBand + 1, uzero, vdim - 1);
			unsigned ws = Clamp((unsigned int)m_min[2] - exactBand, uzero, wdim - 1);
			unsigned we = Clamp((unsigned int)m_max[2] + exactBand + 1, uzero, wdim - 1);
			for (unsigned w = ws; w <= we; ++w) {
				const unsigned wvu = w * vudim;
				for (unsigned v = vs; v <= ve; ++v) {
					const unsigned vd = wvu + v * udim;
					for (unsigned u = us; u <= ue; ++u) {
						unsigned index = vd + u;
						mark[vd + u] = 1;
					}
				}
			}
		});
		m_logger->info("ComputeSeeds Time:{}ms", clock() - st);
		st = clock();
		concurrency::parallel_for(uzero, totalSize, [&](unsigned int i) {
			if (mark[i] == 1) {
				fcpw::Interaction<3> rest;
				m_pTree->findClosestPoint(markpts[i], rest);
				sdfValue[i] = rest.d;
#if CPT_NN_TRIANGLE
				neastTriangle[i] = rest.primitiveIndex;
#endif
			}
		});
		free(mark);
		free(markpts);
		m_logger->info("findClosestPoint Time:{}ms", clock() - st);
		st = clock();
		SolveOnIsotropGrid(udim, vdim, wdim, sample, sample, sample, sample, sdfValue
#if CPT_NN_TRIANGLE
			,pointd,faced,neastTriangle
#endif
		);
		m_logger->info("SolveOnIsotropGrid Time:{}ms", clock() - st);
	}

	namespace
	{
		struct fim_t {
			struct indice_t {
				size_t i, j, k, ind;
				indice_t() {}
				indice_t(size_t ii, size_t ij, size_t ik, size_t iind) : i(ii), j(ij), k(ik), ind(iind) {}
			};
			typedef std::vector< indice_t > active_list_t;
			enum states { source = 0, active = 1, computed, farc };
			std::vector< states > state_cells;
			active_list_t         L;
			fim_t(unsigned ni, unsigned nj, unsigned nk, Real* sol, Real max_float = std::numeric_limits< Real >::max())
				: state_cells(ni * nj * nk, farc), L(), m_dimensions(ni, nj, nk, ni * nj * nk) {
				const unsigned stride_i = 1;
				unsigned       stride_j = ni;
				unsigned       stride_k = ni * nj;
				L.reserve(ni);
				m_active_swapper.reserve(ni);
				for (size_t k = 0; k < nk; ++k) {
					size_t ind_k = k * stride_k;
					for (size_t j = 0; j < nj; ++j) {
						size_t ind_j = ind_k + j * stride_j;
						for (size_t i = 0; i < ni; ++i) {
							size_t ind_i = ind_j + i;
							if (sol[ind_i] < max_float) {
								state_cells[ind_i] = source;
								active_neighbours(i, j, k, ind_i, stride_i, stride_j, stride_k);
							}
						}
					}
				}
			}

			
			Real solveQuadraticEquation(Real u0, Real u1, Real u2, Real inv_f) {
				Real u = u0 + inv_f;
				if (u > u1) {
					u = 0.5f * (u0 + u1 + std::sqrt(-u0 * u0 - u1 * u1 + 2 * u0 * u1 + 2.f * inv_f * inv_f));
				}
				if (u > u2) {
					Real s = u0 + u1 + u2;
					u = (s + std::sqrt( s * s - 3 * (u0 * u0 + u1 * u1 + u2 * u2 - inv_f * inv_f))) / 3.f;
				}
				return u;
			}

			Real solve_hamiltonian_order_1(size_t i, size_t j, size_t k, size_t ind,
				size_t ni, size_t nj, size_t nk, size_t nbcells,
				size_t stride_i, size_t stride_j, size_t stride_k,
				const Real* sol, Real speed)
			{
				size_t ind_left = ind - stride_i;
				size_t ind_right = ind + stride_i;
				size_t ind_top = ind - stride_j;
				size_t ind_bottom = ind + stride_j;
				size_t ind_front = ind - stride_k;
				size_t ind_back = ind + stride_k;

				const Real maxValue = std::numeric_limits<Real>::max();

				Real u_left = (i > 0 ? sol[ind_left] : maxValue);
				Real u_right = (i < ni - 1 ? sol[ind_right] : maxValue);
				Real ulr = std::min(u_left, u_right);

				Real u_top = (j > 0 ? sol[ind_top] : maxValue);
				Real u_bot = (j < nj - 1 ? sol[ind_bottom] : maxValue);
				Real utb = std::min(u_top, u_bot);

				Real u_front = (k > 0 ? sol[ind_front] : maxValue);
				Real u_back = (k < nk - 1 ? sol[ind_back] : maxValue);
				Real ufb = std::min(u_front, u_back);

				if (ulr > utb) std::swap(ulr, utb);
				if (utb > ufb) std::swap(utb, ufb);
				if (ulr > utb) std::swap(ulr, utb);

				return std::min(sol[ind], solveQuadraticEquation(ulr, utb, ufb, 1.f / speed));
			}

			void active_neighbours(size_t i, size_t j, size_t k, size_t ind, size_t stride_i, size_t stride_j,
				size_t stride_k) {
				if (i > 0) {
					size_t  ind_m_i = ind - stride_i;
					states& st = state_cells[ind_m_i];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i - 1, j, k, ind_m_i));
					}
				}
				if (i < m_dimensions.i - 1) {
					size_t  ind_p_i = ind + stride_i;
					states& st = state_cells[ind_p_i];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i + 1, j, k, ind_p_i));
					}
				}
				if (j > 0) {
					size_t  ind_m_j = ind - stride_j;
					states& st = state_cells[ind_m_j];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i, j - 1, k, ind_m_j));
					}
				}
				if (j < m_dimensions.j - 1) {
					size_t  ind_p_j = ind + stride_j;
					states& st = state_cells[ind_p_j];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i, j + 1, k, ind_p_j));
					}
				}
				if (k > 0) {
					size_t  ind_m_k = ind - stride_k;
					states& st = state_cells[ind_m_k];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i, j, k - 1, ind_m_k));
					}
				}
				if (k < m_dimensions.k - 1) {
					size_t  ind_p_k = ind + stride_k;
					states& st = state_cells[ind_p_k];
					if (st == farc) {
						st = active;
						L.push_back(indice_t(i, j, k + 1, ind_p_k));
					}
				}
			}

			void update(Real* sol,
#if CPT_NN_TRIANGLE
				const Vector3*pointD, const int* faced, unsigned int*& neastTriangle,
#endif
				const Real* speed, const Real& eps = 1.E-6) {
				size_t ninj = m_dimensions.i * m_dimensions.j;
				m_active_swapper.resize(0);  
				size_t nL = static_cast<size_t>(L.size());
				for (size_t i = 0; i < nL; ++i) {
					Real p = sol[L[i].ind];
					Real q = solve_hamiltonian_order_1(L[i].i, L[i].j, L[i].k, L[i].ind, m_dimensions.i, m_dimensions.j,
						m_dimensions.k, m_dimensions.ind, 1, m_dimensions.i, ninj, sol,
						(speed == NULL ? 1.f : speed[L[i].ind]));
					sol[L[i].ind] = std::min(q, p);
					if (std::abs(p - q) < eps) {
						indice_t neighbours_ind[6];
						size_t   nb_neighbours = 0;
						if (L[i].i > 0)
							neighbours_ind[nb_neighbours++] = indice_t(L[i].i - 1, L[i].j, L[i].k, L[i].ind - 1);
						if (L[i].i < m_dimensions.i - 1)
							neighbours_ind[nb_neighbours++] = indice_t(L[i].i + 1, L[i].j, L[i].k, L[i].ind + 1);
						if (L[i].j > 0)
							neighbours_ind[nb_neighbours++] =
							indice_t(L[i].i, L[i].j - 1, L[i].k, L[i].ind - m_dimensions.i);
						if (L[i].j < m_dimensions.j - 1)
							neighbours_ind[nb_neighbours++] =
							indice_t(L[i].i, L[i].j + 1, L[i].k, L[i].ind + m_dimensions.i);
						if (L[i].k > 0)
							neighbours_ind[nb_neighbours++] = indice_t(L[i].i, L[i].j, L[i].k - 1, L[i].ind - ninj);
						if (L[i].k < m_dimensions.k - 1)
							neighbours_ind[nb_neighbours++] = indice_t(L[i].i, L[i].j, L[i].k + 1, L[i].ind + ninj);
						for (unsigned n = 0; n < nb_neighbours; ++n) {
							if (state_cells[neighbours_ind[n].ind] == farc) {
								size_t ii, ij, ik, iind;
								ii = neighbours_ind[n].i;
								ij = neighbours_ind[n].j;
								ik = neighbours_ind[n].k;
								iind = neighbours_ind[n].ind;
								Real p = sol[iind];
								Real q = solve_hamiltonian_order_1(ii, ij, ik, iind, m_dimensions.i, m_dimensions.j,
									m_dimensions.k, m_dimensions.ind, 1, m_dimensions.i,
									ninj, sol, (speed == NULL ? 1.f : speed[iind]));
								if (p > q) {
#if CPT_NN_TRIANGLE

#else
									sol[iind] = q;
#endif
									state_cells[iind] = active;
									m_active_swapper.push_back(neighbours_ind[n]);
								}
							}
						}
						state_cells[L[i].ind] = computed;
					}
					else {
						m_active_swapper.push_back(L[i]);
					}
				}  
				m_active_swapper.swap(L);
			}

		private:
			indice_t m_dimensions;
			active_list_t m_active_swapper;
		};

	}


	void OptimizedSdf::SolveOnIsotropGrid(
		unsigned ni, unsigned nj, unsigned nk,
		Real lbx, Real lby, Real lbz,
		Real h, Real* sol, 
#if CPT_NN_TRIANGLE
		const Vector3*pointD, const int* faced, unsigned int*& neastTriangle,
#endif
		Real* speed,Real max_float)
	{
		fim_t fim(ni, nj, nk, sol, max_float);
		while (fim.L.empty() == false)
		{
			fim.update(sol,
#if CPT_NN_TRIANGLE
				pointD,faced, neastTriangle,
#endif
				speed);
		}
	}

	

}
