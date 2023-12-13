// Minimal contour method and implementation by Jozsef Molnar
#pragma once
#include "commontype.h"
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>

#define _VECTSWITCH 1

struct SVeloData
{
	SVeloData(int ix, int iy, double rv):x(ix),y(iy),v(rv) { }
	SVeloData():x(0),y(0),v(0) { }
	int x;
	int y;
	double v;
};

// Abstract base class for shortest path calculation
// The implemented functions are common for all methods

class CEikonal
{
public:
	CEikonal(void);
	~CEikonal(void);

	SWorkImg<double> m_distance;
	SWorkImg<double> m_field;

	void InitImageQuant0(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue);
	void InitImageQuant0(SWorkImg<double> &img);

	virtual void InitImageQuant(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) = 0;
	virtual void InitImageQuant(SWorkImg<double> &img) = 0;
	virtual void DistanceCalculator() = 0;

	virtual void SetDataTerm(SWorkImg<double> *p1, SWorkImg<double> *p2) {}
	virtual void SetDataTerm(SWorkImg<double> *p) {}
	virtual void GetDataTerm(SWorkImg<double> **p1, SWorkImg<double> **p2) {}
	virtual void GetDataTerm(SWorkImg<double> **p) {}

	virtual int SetParam(int p1, int p2) { return 0; }
	virtual int SetParam(int p) { return 0; }
	
	std::vector<CVec2>& ResolvePath();
	int m_resolvready;
	void SetStartStop(const CVec2 &reference,const CVec2 &target);
	int GetProgress() {
		return 100*(m_dstart-m_dcurr)/m_dstart;
	}
	
	virtual void Clean() = 0;

	// Can be called after InitEnvironment(AllMethods) to clean (free memory of) temporal work image containers
	void ResetTempContainers() {
		m_temp[0].Clean();
		m_temp[1].Clean();
		m_aux[0].Clean();
		m_aux[1].Clean();
	}
	void SetBoundaries(int startX, int startY, int endX, int endY){
	    mStartX = startX;
	    mStartY = startY;
	    mEndX = endX;
	    mEndY = endY;
	    GetMaxAuxGrad();
	}
    virtual void CalcImageQuant() = 0;
protected:
	void InitEnvironment(int spacex, int spacey);
	void UpdateDistanceMap(double maxv);
	void GetMaxAuxGrad();
	virtual void GradientCorrection(CVec2 &dir, int x, int y) = 0;

	std::vector<SVeloData> m_velo;
#if _VECTSWITCH
	std::vector<unsigned long> m_auxset;
#else
	std::unordered_set<unsigned long> m_auxset;
#endif
	std::vector<unsigned long> m_boundary;

	double m_currentdistance;
	int m_spacex;
	int m_spacey;

	CVec2 m_reference;
	int m_xdisto;
	int m_ydisto;
	std::vector<CVec2> m_curpath;

	// for "progress"
	int m_dstart;
	int m_dcurr;
	/*CVec2 m_drift;*/

	SWorkImg<double> m_temp[2];
	SWorkImg<double> m_aux[2];
	double m_maxauxgrad;
	double m_minuplevel = 0.35f;

	int m_iDataPrepared;
	int mStartX, mStartY, mEndX, mEndY;
};

enum PrepStat { Prep_No = 0, Prep_Own, Prep_Ext };
// Randers metric specific functions

class CRanders :
	public CEikonal
{
public:
	CRanders();
	~CRanders();

	void DistanceCalculator();
    void InitImageQuantGrad0(SWorkImg<double> &gradx, SWorkImg<double> &grady);
    void InitImageQuantGrad0(const SWorkImg<double> &gradx, const SWorkImg<double> &grady);
	void InitImageQuant(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
		if (m_iDataPrepared == Prep_No) {
			InitImageQuant0(red, green, blue); // new dataterm here
			m_iDataPrepared = Prep_Own;
		}
	}
	void InitImageQuantGrad(SWorkImg<double> &gradx, SWorkImg<double> &grady) {
		if (m_iDataPrepared == Prep_No) {
			InitImageQuantGrad0(gradx, grady); // new dataterm here
			m_iDataPrepared = Prep_Own;
		}
	}
	void InitImageQuant(SWorkImg<double> &img) {
		if (m_iDataPrepared == Prep_No) {
			InitImageQuant0(img); // new dataterm here
			m_iDataPrepared = Prep_Own;
		}
	}

	virtual void GetDataTerm(SWorkImg<double> **p1, SWorkImg<double> **p2) {
		*p1 = m_pTang[0];
		*p2 = m_pTang[1];
	}
	void SetDataTerm(SWorkImg<double> *p1, SWorkImg<double> *p2) {
		if (!p1 || !p2) return;
		Clean();
		m_pTang[0] = p1;
		m_pTang[1] = p2;
		InitEnvironment(p1->xs,p1->ys);
		m_iDataPrepared = Prep_Ext;
	}

	static const int m_expfacini = 8;
	// edge tracking parameter expfac: higher is stronger 
	int SetParam(int expfac = m_expfacini) {
		if (m_expfac == expfac) return 0;
		m_expfac = expfac;
		return 1;
	}
	void Clean() {
		if (m_iDataPrepared == Prep_Own) {
			if (m_pTang[0]) delete m_pTang[0];
			if (m_pTang[1]) delete m_pTang[1];
		}
		m_pTang[0] = 0;
		m_pTang[1] = 0;
		m_iDataPrepared = Prep_No;
	}
    void CalcImageQuant();
protected:
	SWorkImg<double> *m_pTang[2];
	int m_expfac;

	void GradientCorrection(CVec2 &dir, int x, int y) {
		SWorkImg<double> &tangx = *(m_pTang[0]);
		SWorkImg<double> &tangy = *(m_pTang[1]);
        double grad_magnitude = sqrt(tangx[y][x]*tangx[y][x]+tangy[y][x]*tangy[y][x]);
        double exp_mul = (1.0-exp(-grad_magnitude*m_expfac/m_maxauxgrad))/grad_magnitude;

		dir.x += tangx[y][x]*exp_mul; dir.y += tangy[y][x]*exp_mul;
	}
		
};

// Splitter metric specific functions

class CSplitter :
	public CEikonal
{
public:
	CSplitter(void);
	~CSplitter(void);

	void DistanceCalculator();

	void InitImageQuant(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
		if (m_iDataPrepared == Prep_No) {
			InitImageQuant0(red, green, blue); // new dataterm here
			m_iDataPrepared = Prep_Own;
		}
	}
	void InitImageQuant(SWorkImg<double> &img) {
		if (m_iDataPrepared == Prep_No) {
			InitImageQuant0(img); // new dataterm here
			m_iDataPrepared = Prep_Own;
		}
	}

	void GetDataTerm(SWorkImg<double> **p) {
		*p = m_pData;
	}
	void SetDataTerm(SWorkImg<double> *p) {
		if (!p) return;
		Clean();
		m_pData = p;
		InitEnvironment(p->xs,p->ys);
		m_iDataPrepared = Prep_Ext;
	}

	static const int m_expfacini = 3;
	static const int m_relweightpercentini = 70;
	// edge tracking parameter expfac: higher is stronger 
	// tracking vs directed motion (transecting) parameter relweightpercent: higher means stronger transecting (split) power
	int SetParam(int expfac = m_expfacini, int relweightpercent = m_relweightpercentini) {
		if (m_expfac == expfac && abs(m_relweight-relweightpercent/100.0f) < 0.001f) return 0;
		m_expfac = expfac;
		m_relweight = relweightpercent/100.0f;
		return 1; // changed
	}
	void Clean() {
		if (m_iDataPrepared == Prep_Own) {
			if (m_pData) delete m_pData;
		}
		m_pData = 0;
		m_iDataPrepared = Prep_No;
	}
	void CalcImageQuant();
protected:
	SWorkImg<double> *m_pData;
	int m_expfac;
	double m_relweight;

	void GradientCorrection(CVec2 &dir, int x, int y) {
		/**/
		double driftx = (double)(m_xdisto-x), drifty = (double)(m_ydisto-y);
		double dn = 1.0/sqrt(driftx*driftx+drifty*drifty+1e-11);
		driftx *= dn; drifty *= dn;
		dir.x += driftx; dir.y += drifty;
		/**/
		/*dir.x += m_drift.x; dir.y += m_drift.y;*/
	}

};


class CInhomog :
	public CEikonal
{
public:
	CInhomog(void);
	virtual ~CInhomog(void);

	void DistanceCalculator();
	void InitImageQuant(SWorkImg<double>& red, SWorkImg<double>& green, SWorkImg<double>& blue) {
		if (m_iDataPrepared == Prep_No) {
            InitEnvironment(red.xs, red.ys);
            m_aux[0] = red;	m_aux[0] += green; m_aux[0] += blue; m_aux[0] *= 0.333f;
            CalcImageQuant();
			m_iDataPrepared = Prep_Own;
		}
	}
	void InitImageQuant(SWorkImg<double>& img) {
		if (m_iDataPrepared == Prep_No) {
            m_aux[0] = img;
            InitEnvironment(m_aux[0].xs, m_aux[0].ys);
            CalcImageQuant();
			m_iDataPrepared = Prep_Own;
		}
	}

	virtual void GetDataTerm(SWorkImg<double>** p) {
		*p = m_pData;
	}
	void SetDataTerm(SWorkImg<double>* p) {
		if (!p) return;
		Clean();
		m_pData = p;
		InitEnvironment(p->xs, p->ys);
		m_iDataPrepared = Prep_Ext;
	}

	static const int m_expfacini = 8;
	// edge tracking parameter expfac: higher is stronger
	int SetParam(int expfac = m_expfacini) {
		if (m_expfac == expfac) return 0;
		m_expfac = expfac;
		return 1;
	}
	void Clean() {
		if (m_iDataPrepared == Prep_Own) {
			if (m_pData) delete m_pData;
		}
		m_pData = 0;
		m_iDataPrepared = Prep_No;
	}
	void CalcImageQuant();
protected:
	SWorkImg<double>* m_pData;
	int m_expfac;

	void GradientCorrection(CVec2& dir, int x, int y) { }

};

/////////////////////////////////////////////////////
// Control structure (wrapper class for simple use)
/////////////////////////////////////////////////////

struct SControl 
{
	SControl() {
		m_pMethods[0] = &m_Randers;
		m_pMethods[1] = &m_Splitter;
		m_pMethods[2] = &m_Inhomog;
		m_pCurrentMethod = 0;
		m_iParaToUpdate = 0;
		m_pdats = m_pdatr[0] = m_pdatr[1] = 0;
	}
	~SControl() {
		Clean();
	}
	void Clean() {
		if (m_pCurrentMethod)
			m_pCurrentMethod->Clean();
	}

	void CleanAll() {
		m_Randers.Clean();
		m_Splitter.Clean();
		m_Inhomog.Clean();
	}

	void SetBoundaries(int startX, int startY, int endX, int endY){
	    m_Randers.SetBoundaries(startX,startY,endX, endY);
		m_Splitter.SetBoundaries(startX,startY,endX, endY);
		m_Inhomog.SetBoundaries(startX,startY,endX, endY);
	}
	// Prepare all data terms from color image (for sequential use)
	void InitEnvironmentAllMethods(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
		SetParAll();
		m_Randers.InitImageQuant(red,green,blue);
		m_Splitter.InitImageQuant(red,green,blue);
		m_Inhomog.InitImageQuant(red, green, blue);
	}

	void InitEnvironmentRanders(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
	    m_Randers.InitImageQuant(red,green,blue);
	}
	void InitEnvironmentRandersGrad(SWorkImg<double> &gradx, SWorkImg<double> &grady) {
        m_Randers.InitImageQuantGrad(gradx, grady);
    }
	void InitEnvironmentSplitter(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
	    m_Splitter.InitImageQuant(red,green,blue);
	}
	void InitEnvironmentInhomog(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
	    m_Inhomog.InitImageQuant(red,green,blue);
	}

	void InitEnvironmentAllMethods(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue, SWorkImg<double> &gradx, SWorkImg<double> &grady) {
		SetParAll();
		m_Randers.InitImageQuantGrad(gradx, grady);
//		m_Splitter.InitImageQuant(red,green,blue);
//		m_Inhomog.InitImageQuant(red, green, blue);
	}

	// Prepare all data terms from grayscale image (for sequential use)
	void InitEnvironmentAllMethods(SWorkImg<double> &img) {
		SetParAll();
		m_Randers.InitImageQuant(img);
		m_Splitter.InitImageQuant(img);
		m_Inhomog.InitImageQuant(img);
	}

	void CalcImageQuantAllMethods(){
	    m_Randers.CalcImageQuant();
		m_Splitter.CalcImageQuant();
		m_Inhomog.CalcImageQuant();
	}
	// Prepare data term from color image (parallel use)
	void InitEnvironment(SWorkImg<double> &red, SWorkImg<double> &green, SWorkImg<double> &blue) {
		SetMetDatPar();
		m_pCurrentMethod->InitImageQuant(red,green,blue);
	}
	// Prepare data term from grayscale image (parallel use)
	void InitEnvironment(SWorkImg<double> &img) {
		SetMetDatPar();
		m_pCurrentMethod->InitImageQuant(img);
	}
	// input: user-defined point set, method: user-defined method set
	bool DefineInputSet(const std::vector<CVec2> &input, const std::vector<int> &method) {
		bool bok = true;
		m_curri = 0;
		m_inputset = input;
		m_pEikonal.clear();
		auto ms = method.size();
		auto is = input.size();
		if (ms != is)
			for (int ii = 0; ii < is; ++ii) {
				m_pEikonal.push_back(m_pMethods[0]); // let it be the default
				bok = false;
			}
		else
			for (int ii = 0; ii < is; ++ii) {
				int methi = method[ii];
				if (methi >= m_nimplenented) {
					methi = 0; // let it be the default
					bok = false;
				}
				m_pEikonal.push_back(m_pMethods[methi]);
			}

		m_minpath.clear();
		return bok;
	}
	// Sets the next data-pairs (from the user input set) for the segment calculations
	int SetNextStartStop() {
		auto ninp = m_inputset.size();
		if (m_curri >= ninp) return 0; // no more points in the input set
		
		m_pCurrentMethod = m_pEikonal[m_curri]; // double setting
		CVec2 reference = m_inputset[m_curri];
		CVec2 target;
		if (++m_curri == ninp)
			target = m_inputset[0];
		else
			target = m_inputset[m_curri];
		m_pCurrentMethod->SetStartStop(reference,target);

		return 1;
	}
	// Main iteration
	void DistanceCalculator() {
		if (&m_Randers == m_pCurrentMethod) {
			m_Randers.DistanceCalculator();
		}
		else if (&m_Splitter == m_pCurrentMethod) {
			m_Splitter.DistanceCalculator();
		}
		else {
		    m_Inhomog.DistanceCalculator();
		}
		m_resolvready = m_pCurrentMethod->m_resolvready;
	}
	// Method parameters
	void SetParam(int p) {
		m_rp = p;
		m_iParaToUpdate |= 1;
	}
	void SetParam(int p1, int p2) {
		m_sp1 = p1; m_sp2 = p2;
		m_iParaToUpdate |= 2;
	}

	// To attach existing data term
	void SetDataTerm(SWorkImg<double> *p) {
		m_pdats = p;
	}
	void SetDataTerm(SWorkImg<double> *p1, SWorkImg<double> *p2) {
		m_pdatr[0] = p1;
		m_pdatr[1] = p2;
	}
	// Queries
	// Retrieve existing data term
	void GetDataTerm(SWorkImg<double> **p) {
		if (m_pCurrentMethod)
			m_pCurrentMethod->GetDataTerm(p);
	}
	void GetDataTerm(SWorkImg<double> **p1, SWorkImg<double> **p2) {
		if (m_pCurrentMethod)
			m_pCurrentMethod->GetDataTerm(p1,p2);
	}

	SWorkImg<double> &GetField() {
		return m_pCurrentMethod->m_field;
	}

	int GetReady() {
		return m_resolvready;
	}

	int GetProgress() {
		return m_pCurrentMethod->GetProgress();
	}
	// Call after each segment is ready
	std::vector<CVec2> &ResolvePath() {
		
		std::vector<CVec2> curpath = m_pCurrentMethod->ResolvePath();
		m_minpath.insert(m_minpath.begin(),curpath.begin(),curpath.end());

		return m_minpath;
	}
	// Retrieve the result
	std::vector<CVec2> &GetMinPath() {
		return m_minpath;
	}
	void SetParAll() {
		if (m_iParaToUpdate&1) {
			m_iParaToUpdate &= ~1;
			if (m_Randers.SetParam(m_rp))
				m_Randers.Clean();
            if (m_Inhomog.SetParam(m_rp))
				m_Inhomog.Clean();
		}
		if (m_iParaToUpdate&2) {
			m_iParaToUpdate &= ~2;
			if (m_Splitter.SetParam(m_sp1, m_sp2))
				m_Splitter.Clean();
		}
	}
	
private:
	void SetMetDatPar() {
		m_pCurrentMethod = m_pEikonal[m_curri];
		m_pCurrentMethod->SetDataTerm(m_pdats);
		m_pCurrentMethod->SetDataTerm(m_pdatr[0],m_pdatr[1]);

		if (m_iParaToUpdate&1) {
			m_iParaToUpdate &= ~1;
			if (m_pCurrentMethod->SetParam(m_rp))
				m_pCurrentMethod->Clean();
		}
		if (m_iParaToUpdate&2) {
			m_iParaToUpdate &= ~2;
			if (m_pCurrentMethod->SetParam(m_sp1, m_sp2))
				m_pCurrentMethod->Clean();
		}
	}
	static const int m_nimplenented = 3; // # of implemented methods

	std::vector<CVec2> m_minpath;
	int m_resolvready;
	CEikonal *m_pMethods[m_nimplenented];
	CEikonal *m_pCurrentMethod;
	CRanders m_Randers;
	CSplitter m_Splitter;
	CInhomog m_Inhomog;

	std::vector<CVec2> m_inputset;
	std::vector<CEikonal *> m_pEikonal;
	int m_curri;
	int m_iParaToUpdate;
	int m_rp;
	int m_sp1;
	int m_sp2;
	SWorkImg<double> *m_pdats;
	SWorkImg<double> *m_pdatr[2];
};