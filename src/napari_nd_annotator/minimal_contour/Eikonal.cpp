#include "Eikonal.h"
#include <vector>
#include <atomic>
#include "commontype.h"

using namespace std;

CEikonal::CEikonal(void):m_spacex(0),m_spacey(0),m_iDataPrepared(Prep_No)
{
}

CEikonal::~CEikonal(void)
{
}

////////////////////////////////////////////////////////////////////////
// Containers: m_distance: distance map, field: container for subpixel
////////////////////////////////////////////////////////////////////////

void CEikonal::InitEnvironment(int spacex, int spacey)
{
	m_distance.Set(spacex,spacey,-1);
	m_field.Set(spacex,spacey,-1);
	m_velo.reserve(spacex*spacey);
	m_boundary.reserve(spacex*spacey);

	m_spacex = spacex;
	m_spacey = spacey;}

////////////////////////////////
// Receiving Start-Stop points
////////////////////////////////


void CEikonal::SetStartStop(const CVec2 &reference,const CVec2 &target)
{

	m_reference = reference;
	
	m_distance.Set(m_spacex,m_spacey,-1);
	m_field.Set(m_spacex,m_spacey,-1);

	int cx = (int)m_reference.x, cy = (int)m_reference.y;

	m_currentdistance = 3.0f;
	for (int yy = 0; yy < m_spacey; ++yy) {
		for (int xx = 0; xx < m_spacex; ++xx) {
			int dx = xx-cx, dy = yy-cy;
			double dd = sqrt((double)(dx*dx+dy*dy));
			
			if (dd < m_currentdistance) {
				m_field[yy][xx] = 1.0-2.0f*dd/m_currentdistance;
				m_distance[yy][xx] = dd;
			}
			else {
				m_field[yy][xx] = -1;
				m_distance[yy][xx] = -1;
			}

		}
	}

	m_boundary.clear();

	for (int yy = 1; yy < m_spacey - 1; ++yy)
		for (int xx = 1; xx < m_spacex - 1; ++xx)
			if (m_field[yy][xx] < m_minuplevel) {
				if (m_distance[yy + 1][xx] >= 0 || m_distance[yy - 1][xx] >= 0
				 || m_distance[yy][xx + 1] >= 0 || m_distance[yy][xx - 1] >= 0) {
					m_boundary.push_back(yy*0x10000+xx);
				}
			}


	m_xdisto = (int)target.x;
	m_ydisto = (int)target.y;

	m_resolvready = 30; // damping;

	// "progress"
	m_dcurr = m_dstart = 1+abs(m_xdisto-cx)+abs(m_ydisto-cy);
	/*m_drift = target;
	m_drift -= m_reference;
	if (m_drift.Len() > 1e-11) m_drift.Norm();*/
}

//////////////////////////////////////////////////
// Image to Control Data preparation common part
//////////////////////////////////////////////////

void CEikonal::GetMaxAuxGrad()
{
	int ys = m_aux[0].ys, xs = m_aux[0].xs;
	m_maxauxgrad = 0;
	for (int yy = 0; yy < ys; ++yy) {
		for (int xx = 0; xx < xs; ++xx) {
			double sq = m_aux[0][yy][xx]*m_aux[0][yy][xx]+m_aux[1][yy][xx]*m_aux[1][yy][xx];
			if (sq > m_maxauxgrad) m_maxauxgrad = sq;
		}
	}
	m_maxauxgrad = sqrt(m_maxauxgrad);
}

void CEikonal::InitImageQuant0(SWorkImg<double> &img)
{
	InitEnvironment(img.xs,img.ys);

	img.GetImgGrad(m_aux[1],m_aux[0]);
	m_aux[1] *= -1;
	GetMaxAuxGrad();
	CalcImageQuant();
}

void CEikonal::InitImageQuant0(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue)
{
	if (green.xs != red.xs || blue.xs != red.xs) return;
	if (green.ys != red.ys || blue.ys != red.ys) return;
	int ys = red.ys, xs = red.xs;
	InitEnvironment(xs,ys);
	red.GetImgGrad(m_aux[1],m_aux[0]);
	green.GetImgGrad(m_temp[1],m_temp[0]);
	for (int yy = 0; yy < ys; ++yy) {
		for (int xx = 0; xx < xs; ++xx) {
			m_aux[0][yy][xx] += m_temp[0][yy][xx];
			m_aux[1][yy][xx] += m_temp[1][yy][xx];
		}
	}
	blue.GetImgGrad(m_temp[1],m_temp[0]);
	for (int yy = 0; yy < ys; ++yy) {
		for (int xx = 0; xx < xs; ++xx) {
			m_aux[0][yy][xx] += m_temp[0][yy][xx];
			m_aux[1][yy][xx] += m_temp[1][yy][xx];
		}
	}

	m_aux[1] *= -1;
	GetMaxAuxGrad();
	CalcImageQuant();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Update the distance map by values from DistanceCalculator-s called from specific DistanceCalculator-s
//////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void CEikonal::UpdateDistanceMap(double maxv)
{

	if (maxv < 1e-22f) maxv += 1e-22f;
	double mdatspedmax = 2+0.5f;//-2
	maxv = mdatspedmax/maxv;
	m_currentdistance += maxv;

	auto si = m_velo.size();
	if (!si) { // complete distance map 
		m_resolvready = -1;
		return;
	}

	int mind(100000000);

	int xs = m_field.xs, ys = m_field.ys;
	m_auxset.clear();
	m_boundary.clear();

	for (int ii = 0; ii < si; ++ii) {
		SVeloData& sv = m_velo[ii];
		int xx = sv.x, yy = sv.y;

		// "progress"
		int ad = abs(m_xdisto-xx)+abs(m_ydisto-yy);
		if (mind > ad) mind = ad;

		double vv = sv.v*maxv;
		m_field[yy][xx] += vv;

		if (m_field[yy][xx] > 0 && m_distance[yy][xx] < -0.5f) {
			m_distance[yy][xx] = m_currentdistance;
		
			int iy = yy, ix = xx;
			++ix;
			if (ix > 0 && ix < xs - 1 && iy > 0 && iy < ys - 1)
#if _VECTSWITCH
				m_auxset.emplace_back(iy * 0x10000 + ix);
#else
				m_auxset.emplace(iy * 0x10000 + ix);
#endif
			ix -= 2;
			if (ix > 0 && ix < xs - 1 && iy > 0 && iy < ys - 1)
#if _VECTSWITCH
				m_auxset.emplace_back(iy * 0x10000 + ix);
#else
				m_auxset.emplace(iy * 0x10000 + ix);
#endif
			++ix; ++iy;
			if (ix > 0 && ix < xs - 1 && iy > 0 && iy < ys - 1)
#if _VECTSWITCH
				m_auxset.emplace_back(iy * 0x10000 + ix);
#else
				m_auxset.emplace(iy * 0x10000 + ix);
#endif
			iy -= 2;
			if (ix > 0 && ix < xs - 1 && iy > 0 && iy < ys - 1)
#if _VECTSWITCH
				m_auxset.emplace_back(iy * 0x10000 + ix);
#else
				m_auxset.emplace(iy * 0x10000 + ix);
#endif
		}
		else 
			if (m_field[yy][xx] < m_minuplevel) m_boundary.push_back(yy * 0x10000 + xx);

	}

	auto it = m_auxset.cbegin(), ite = m_auxset.cend();

	for (; it != ite; ++it) {
		unsigned long cxy = *it;
		int xx = cxy & 0xffff, yy = cxy >> 16;
		if (m_field[yy][xx] < m_minuplevel)
			if (m_distance[yy + 1][xx] >= 0 || m_distance[yy - 1][xx] >= 0
				|| m_distance[yy][xx + 1] >= 0 || m_distance[yy][xx - 1] >= 0)
					m_boundary.push_back(cxy);
	}

	if (m_distance[m_ydisto][m_xdisto] > 0) // distance map built to the point it has to
		if (m_resolvready > 0) --m_resolvready;

	// "progress"
	m_dcurr = mind;

}

////////////////////////////////////
// Shortest path from distance map
////////////////////////////////////
#if 0
std::vector<CVec2>& CEikonal::ResolvePath()
{
	SWorkImg<double> &distance = m_distance;
	int xs = distance.xs, ys = distance.ys;
	m_curpath.clear();

	int ix(m_xdisto), iy(m_ydisto);

	if (ix < 1) ix = 1;
	else if (ix >= xs-1) ix = xs-2;
	if (iy < 1) iy = 1;
	else if (iy >= ys-1) iy = ys-2;

	double sky = 1.2f*m_currentdistance;

	CVec2 path((double)ix,(double)iy);
	int lx = -1, ly = -1;
	m_curpath.push_back(path);

	for (int ii = 0; ii < 4444; ++ii) {
		CVec2 dir;
		ix = (int)(path.x+0.0f); iy = (int)(path.y+0.0f);
		double dxp = distance[iy][ix+1]; if (dxp < 0) dxp = sky;
		double dxm = distance[iy][ix-1]; if (dxm < 0) dxm = sky;
		double dyp = distance[iy+1][ix]; if (dyp < 0) dyp = sky;
		double dym = distance[iy-1][ix]; if (dym < 0) dym = sky;
		double gradx = 0.5f*(dxp-dxm), grady = 0.5f*(dyp-dym);

		dir.x = gradx; dir.y = grady;
		GradientCorrection(dir,ix,iy); // specific call

		dir.Norm();
		path -= dir;
		
		int px = (int)(path.x+0.49f), py = (int)(path.y+0.49f);
		if (lx != px || ly != py) {
			m_curpath.push_back(CVec2((double)px,(double)py));
			lx = px; ly = py;
		}

		dir = path; dir -= m_reference;
		if (dir.x*dir.x+dir.y*dir.y < 2.0f) {
			m_curpath.push_back(m_reference);
			break;
		}
	}

	return m_curpath;

}
#else
std::vector<CVec2>& CEikonal::ResolvePath()
{
	SWorkImg<double>& distance = m_distance;
	int xs = distance.xs, ys = distance.ys;
	double qc = 1.0;

repall:
	m_curpath.clear();

	int ix(m_xdisto), iy(m_ydisto);
	if (ix < 1) ix = 1;
	else if (ix >= xs - 1) ix = xs - 2;
	if (iy < 1) iy = 1;
	else if (iy >= ys - 1) iy = ys - 2;

	double sky = 1.2f * m_currentdistance;

	CVec2 path((double)ix, (double)iy);
	m_curpath.push_back(path);


	int iimax = 4444, ii;
	for (ii = 0; ii < iimax; ++ii) {
		CVec2 dir;
		int mx(0), my(0);

		dir = m_reference; dir -= path;
		if (dir.x * dir.x + dir.y * dir.y < 2.0f) {
			m_curpath.push_back(m_reference);
			break;
		}

		ix = (int)(path.x+0.0001f); iy = (int)(path.y+0.0001f);
		dir.x = 0.0f; dir.y = 0.0f;
		GradientCorrection(dir, ix, iy); // specific call (additive)
		dir *= qc;

		double minv(0);

		int frm = 1;
reit:		
		for (int dy = -frm; dy <= frm; ++dy) {
			for (int dx = -frm; dx <= frm; ++dx) {
				if (!dx && !dy) continue;
				double idlen = 1.0 / sqrt((double)(dx * dx + dy * dy));
				double dd = sky;
				int px = ix + dx, py = iy + dy;
				if (px >= 1 && px < xs-1 && py >= 1 && py < ys-1)
					if (distance[py][px] >= 0) dd = distance[py][px];

				dd -= distance[iy][ix] >= 0 ? distance[iy][ix] : sky;
				dd += dir.x * dx + dir.y * dy;
				dd *= idlen;
				if (dd < minv) {
					minv = dd;
					mx = dx; my = dy;
				}
			}
		}
		if (!mx && !my) { // should not happen
			if (++frm <= 2) goto reit;
		}

		path = CVec2((double)(ix + mx), (double)(iy + my));
		m_curpath.push_back(path);

	}
	if (ii == iimax) {
		if (qc > 0.75f) {
			qc = 0.5f;
			goto repall;
		}
	}

	return m_curpath;

}
#endif
//////////////////////////////////////////////////////////////
// CRanders: Implementation of the edge based Randers metric
//////////////////////////////////////////////////////////////

CRanders::CRanders()
{
	m_pTang[0] = m_pTang[1] = 0;
	SetParam();
}

CRanders::~CRanders()
{
	 Clean();
}

void CRanders::DistanceCalculator()
{
	SWorkImg<double> &tangx = *(m_pTang[0]);
	SWorkImg<double> &tangy = *(m_pTang[1]);

	int xs = m_field.xs, ys = m_field.ys;

	// evolution logic

	double maxv(0);
	m_velo.clear();

	auto bsi = m_boundary.size();
	for (int ii = 0; ii < bsi ; ++ii) {

		unsigned long cxy = m_boundary[ii];
		int xx = cxy & 0xffff, yy = cxy >> 16;

		double nx = 0.5f * (m_field[yy][xx + 1] - m_field[yy][xx - 1]);
		double ny = 0.5f * (m_field[yy + 1][xx] - m_field[yy - 1][xx]);
		double gradlen = sqrt(nx * nx + ny * ny);
		if (gradlen < 1e-11) gradlen = 1e-11;
		double igradlen = 1.0 / gradlen;
		nx *= igradlen; ny *= igradlen;

		double Qn = tangx[yy][xx] * nx + tangy[yy][xx] * ny;
		double Qt2 = -tangx[yy][xx] * ny + tangy[yy][xx] * nx;
		Qt2 *= Qt2;
		double eikon(1 - Qt2);
		if (eikon < 0) eikon = 0;
		eikon = Qn + sqrt(eikon);
		double vnormed = gradlen / eikon;

		if (vnormed < 1e-9f) vnormed = 1e-9f; // causality criterion
		SVeloData sv(xx, yy, vnormed);

		if (maxv < vnormed) maxv = vnormed;
		m_velo.push_back(sv);		
	}


	// update distance map

	UpdateDistanceMap(maxv);
	
}

////////////////////////////////////////////////////
// Image to Control Data preparation specific part
////////////////////////////////////////////////////

void CRanders::CalcImageQuant()
{
	double maxgrab = 0;

	if (!m_pTang[0]) m_pTang[0] = new SWorkImg<double>;
	if (!m_pTang[0]) return;
	if (!m_pTang[1]) m_pTang[1] = new SWorkImg<double>;
	if (!m_pTang[1]) return;

	*(m_pTang[0]) = m_aux[0];
	*(m_pTang[1]) = m_aux[1]; // check
	if (m_maxauxgrad < 1e-11) return; // essentially Euclidean space
	int xs = m_pTang[0]->xs, ys = m_pTang[0]->ys;
	SWorkImg<double> &tang0 = *(m_pTang[0]);
	SWorkImg<double> &tang1 = *(m_pTang[1]);

	maxgrab = m_expfac/m_maxauxgrad; // 0 - 8

	for (int yy = 0; yy < ys; ++yy) {
		for (int xx = 0; xx < xs; ++xx) {
			double q = tang0[yy][xx]*tang0[yy][xx]+tang1[yy][xx]*tang1[yy][xx];
			q = sqrt(q);
			if (q > 1e-11) {
				tang0[yy][xx] /= q; tang1[yy][xx] /= q;
				q *= maxgrab;
				q = 1.0-((double)exp(-q));	// ~ 0 - 0.632
				tang0[yy][xx] *= q; tang1[yy][xx] *= q;
			}
			else {
				tang0[yy][xx] = tang1[yy][xx] = 0;
			}
		}
	}

}

////////////////////////////////////////////////////////////////
// CSplitter: Implementation of "splitting along edges" metric
////////////////////////////////////////////////////////////////

CSplitter::CSplitter(void)
{
	m_pData = 0;
	SetParam();
}

CSplitter::~CSplitter(void)
{
	Clean();
}

void CSplitter::DistanceCalculator()
{
	SWorkImg<double> &data = *m_pData;

	int xs = m_field.xs, ys = m_field.ys;

	// evolution logic

	double maxv(0);
	m_velo.clear();
	int tx = m_xdisto, ty = m_ydisto;

	auto bsi = m_boundary.size();
	for (int ii = 0; ii < bsi; ++ii) {

		unsigned long cxy = m_boundary[ii];
		int xx = cxy & 0xffff, yy = cxy >> 16;

		double nx = 0.5f*(m_field[yy][xx+1]-m_field[yy][xx-1]);
		double ny = 0.5f*(m_field[yy+1][xx]-m_field[yy-1][xx]);
		double gradlen = sqrt(nx*nx+ny*ny);
		if (gradlen < 1e-11) gradlen = 1e-11;
		double igradlen = 1.0/gradlen;
		nx *= igradlen; ny *= igradlen;

		/**/
		double driftx = (double)(tx-xx), drifty = (double)(ty-yy);
		double dn = 1.0/sqrt(driftx*driftx+drifty*drifty+1e-11);
		driftx *= dn; drifty *= dn;
		double Qn = driftx*nx+drifty*ny;
		/**/
		/*double Qn = m_drift.x*nx+m_drift.y*ny;*/

		double eikon = Qn*Qn+data[yy][xx]; 
		if (eikon < 0) eikon = 0;
		eikon = Qn+sqrt(eikon);
					
		double vnormed = gradlen/eikon;
		if (vnormed < 1e-9f) vnormed = 1e-9f; // causality criterion
					
		SVeloData sv(xx,yy,vnormed);

		if (maxv < vnormed) maxv = vnormed;
		m_velo.push_back(sv);

	}


	// update distance map

	UpdateDistanceMap(maxv);
	
}

////////////////////////////////////////////////////
// Image to Control Data preparation specific part
////////////////////////////////////////////////////

void CSplitter::CalcImageQuant()
{
	double maxgrab = m_maxauxgrad;
	if (maxgrab < 1e-11) maxgrab = 1e-11;

	int xs = m_aux[0].xs, ys = m_aux[0].ys;
	if (!m_pData) m_pData = new SWorkImg<double>;
	if (!m_pData) return;	
	m_pData->Set(xs,ys); // check
	SWorkImg<double> &data = *m_pData;

	for (int yy = 0; yy < ys; ++yy) {
		for (int xx = 0; xx < xs; ++xx) {
			double q = m_aux[0][yy][xx]*m_aux[0][yy][xx]+m_aux[1][yy][xx]*m_aux[1][yy][xx];
			q = sqrt(q)/maxgrab; // 0-1
			// edge data
			double dat = m_relweight+(1.0-m_relweight)*((double)exp(-m_expfac*q));
			// use of data for prepared data FI*FI-Q*Q; |Q| := alpha
			data[yy][xx] = dat*dat-m_relweight*m_relweight;
		}
	}

}
