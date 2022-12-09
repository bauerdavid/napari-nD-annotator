#pragma once
#include "math.h"

typedef unsigned char byte;

#define PI 3.1415926536

class CVec2
{
public:
    CVec2() {};
    CVec2(double x, double y)
    {
        v[0] = x;
        v[1] = y;
    };
    ~CVec2() {};

    union
    {
        double v[2];
        struct
        {
            double x, y;
        };
    };

    double& operator[](int i)
    {
        return v[i];
    }
    CVec2& operator*=(double s)
    {
        v[0] *= s;
        v[1] *= s;
        return *this;
    }
    CVec2& operator+=(CVec2& w)
    {
        v[0] += w[0];
        v[1] += w[1];
        return *this;
    }
    CVec2& operator-=(CVec2& w)
    {
        v[0] -= w[0];
        v[1] -= w[1];
        return *this;
    }

    CVec2& Norm()
    {
        double l = 1.0f / sqrt(v[0] * v[0] + v[1] * v[1]);
        v[0] *= l;
        v[1] *= l;
        return *this;
    }

    double Len()
    {
        return sqrt(v[0] * v[0] + v[1] * v[1]);
    }

};
CVec2 operator *(double f, CVec2& s);
CVec2 operator *(int f, CVec2& s);
double operator *(CVec2& s1, CVec2& s2);

class CVec3
{
public:
    CVec3()
    {
        v[0] = v[1] = v[2] = 0;
    };
    CVec3(double x, double y, double z)
    {
        v[0] = x;
        v[1] = y;
        v[2] = z;
    };
    ~CVec3() {};

    union
    {
        double v[3];
        struct
        {
            double x, y, z;
        };
    };

    double& operator[](int i)
    {
        return v[i];
    }
    CVec3& operator*=(double s)
    {
        v[0] *= s;
        v[1] *= s;
        v[2] *= s;
        return *this;
    }
    CVec3& operator+=(CVec3& w)
    {
        v[0] += w[0];
        v[1] += w[1];
        v[2] += w[2];
        return *this;
    }
    CVec3& operator-=(CVec3& w)
    {
        v[0] -= w[0];
        v[1] -= w[1];
        v[2] -= w[2];
        return *this;
    }

    CVec3& Norm()
    {
        double l = 1.0 / sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[3]);
        v[0] *= l;
        v[1] *= l;
        v[2] *= l;
        return *this;
    }

    double Len()
    {
        return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[3]);
    }

};

CVec3 operator *(double f, CVec3& v);

class CMat3
{
public:
    CMat3()
    {
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                m[i][j] = 0;
    };

    ~CMat3() {};

    union
    {
        double m[3][3];
        struct
        {
            CVec3 r[3];
        };
    };

    void SetId()
    {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1;
    }

    CVec3& operator[](int i)
    {
        return r[i];
    }
    CMat3& operator=(const CMat3& s)
    {
        m[0][0] = s.m[0][0]; m[0][1] = s.m[0][1]; m[0][2] = s.m[0][2];
        m[1][0] = s.m[1][0]; m[1][1] = s.m[1][1]; m[1][2] = s.m[1][2];
        m[2][0] = s.m[2][0]; m[2][1] = s.m[2][1]; m[2][2] = s.m[2][2];
        return *this;
    }
    CMat3& operator*=(double s)
    {
        r[0] *= s;
        r[1] *= s;
        r[2] *= s;
        return *this;
    }
    CMat3& operator+=(CMat3& w)
    {
        r[0] += w[0];
        r[1] += w[1];
        r[2] += w[2];
        return *this;
    }
    CMat3& operator-=(CMat3& w)
    {
        r[0] -= w[0];
        r[1] -= w[1];
        r[2] -= w[2];
        return *this;
    }

    CMat3& operator*=(CMat3& w)
    {
        CMat3 tm;
        for(int i = 0; i < 3; ++i)
            for(int j = 0; j < 3; ++j)
                for(int k = 0; k < 3; ++k)
                    tm[i][j] += m[i][k] * w[k][j];

        *this = tm;
        return *this;
    }
    CMat3& Transpose()
    {
        double a;
        a = m[0][1]; m[0][1] = m[1][0]; m[1][0] = a;
        a = m[0][2]; m[0][2] = m[2][0]; m[2][0] = a;
        a = m[1][2]; m[1][2] = m[2][1]; m[2][1] = a;
        return *this;
    }

};

/*CMat3 operator *(double f, CMat3 &m);
CMat3 operator *(CMat3 &m1, CMat3 &m2);
CVec3 operator *(CMat3 &m, CVec3 &v);
CVec3 operator *(CVec3 &v, CMat3 &m);*/

///////////////////////////////////////////////////////////////////////////////
#define HFRAME 8

struct SDisImg
{
    SDisImg()
    {
        dat = 0;
        xs = ys = 0;
    }
    SDisImg(int x, int y, byte* buf, int mod = 0, int pitch = 0)
    {
        dat = 0;
        xs = ys = 0;
        Set(x, y, buf, mod, pitch);
    }
    void Reset(int x, int y)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new unsigned long[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] = 0;
        }
    }
    void Set(int x, int y, byte* buf, int mod = 0, int pitch = 0)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new unsigned long[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                byte r, g, b;
                r = g = b = *buf++;
                dat[q * xs + p] = (r << 16) + (g << 8) + b;
            }
            buf += mod;
            if(pitch < 0) buf += pitch * 2;
        }
    }
    // kaggle
    void InitKaggleSaveImg(int x, int y)
    {
        xs = x;
        ys = y;
        dat = new unsigned long[ys * xs];
        if(!dat)
        {
            xs = ys = 0;
            return;
        }
        for(int q = 0; q < ys; ++q)
            for(int p = 0; p < xs; ++p)
                dat[q * ys + p] = 0;
    }
    void Set32Kaggle(int x, int y, byte* buf, int mod = 0, int pitch = 0)
    {

        if(dat && (x != xs || y != ys))
            Clean();
        xs = x + HFRAME * 2;
        ys = y + HFRAME * 2;
        if(!dat && xs && ys)
        {
            dat = new unsigned long[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        int avg = 0;
        for(int q = 0 + HFRAME; q < ys - HFRAME; ++q)
        {
            for(int p = 0 + HFRAME; p < xs - HFRAME; ++p)
            {
                double gray;
                byte r, g, b;
                r = *buf++;
                g = *buf++;
                b = *buf++;
                gray = 0.3f * r + 0.5f * g + 0.2f * b;
                if(gray > 255) gray = 255;
                r = g = b = byte(gray);
                dat[q * xs + p] = (r << 16) + (g << 8) + b;
                avg += b;
                buf++;
            }
            buf += mod;
            if(pitch < 0) buf += pitch * 2;
        }
        avg /= (xs - HFRAME * 2) * (ys - HFRAME * 2);
        int hist[256];
        for(int i = 0; i < 256; ++i) hist[i] = 0;
        for(int q = 0 + HFRAME; q < ys - HFRAME; ++q)
            for(int p = 0 + HFRAME; p < xs - HFRAME; ++p)
                hist[dat[q * xs + p] & 0xff]++;
        int max = 0; byte mi = 0;
        for(int i = 0; i < 256; ++i)
        {
            if(hist[i] > max) { max = hist[i]; mi = i; }
        }
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                if(p >= HFRAME && p < xs - HFRAME && q >= HFRAME && q < ys - HFRAME) continue;
                dat[q * xs + p] = mi;
            }
        }
        if(mi > avg)
        {
#pragma omp parallel for
            for(int qp =0; qp < ys*xs; qp++){
                dat[qp] = 255 - dat[qp];
            }
        }


    }
    // kaggle
    void SetColor(int x, int y, byte* buf, int mod = 0, int pitch = 0)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new unsigned long[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                byte r, g, b;
                r = *buf++;
                g = *buf++;
                b = *buf++;
                dat[q * xs + p] = (r << 16) + (g << 8) + b;
            }
            buf += mod;
            if(pitch < 0) buf += pitch * 2;
        }
    }
    void Clean()
    {
        if(dat)
        {
            delete[] dat;
            dat = 0;
            xs = ys = 0;
        }
    }
    ~SDisImg()
    {
        Clean();
    }
    unsigned long* operator[](int y)
    {
        if(y >= ys) y = ys - 1;
        else if(y < 0) y = 0;
        return &dat[y * xs];
    }
    void operator=(SDisImg& tc)
    {
        if(&tc == this) return;
        if(xs != tc.xs || ys != tc.ys)
        {
            Clean();
            xs = tc.xs;
            ys = tc.ys;
            dat = new unsigned long[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            dat[qp] = tc[q][p];
        }
    }
    unsigned long* dat;
    int xs;
    int ys;
};

struct SSavImg
{
    SSavImg()
    {
        dat = 0;
        xs = ys = 0;
    }
    ~SSavImg()
    {
        if(dat)
        {
            delete[] dat;
            dat = 0;
        }
    }

    unsigned char* dat;
    int xs;
    int ys;
};

// working with real numbers

template <class T> struct SWorkImg
{
    SWorkImg()
    {
        dat = 0;
        xs = ys = 0;
    }
    SWorkImg(int x, int y)
    {
        dat = 0;
        xs = ys = 0;
        Set(x, y);
    }
    SWorkImg(int x, int y, byte* buf)
    {
        dat = 0;
        xs = ys = 0;
        Set(x, y, buf);
    }
    void Set(int x, int y)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        maxval = minval = avgval = 0;
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] = (T)0;
        }
    }
    void Set(int x, int y, T fill)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        maxval = minval = avgval = 0;
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] = fill;
        }
    }
    void Set(int x, int y, byte* buf)
    {
        if(dat && (x != xs || y != ys))
            Clean();
        xs = x;
        ys = y;
        if(!dat && xs && ys)
        {
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }
        maxval = 0;
        minval = (T)10000;
        avgval = 0;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                T t = (T)buf[qp];
                dat[qp] = t;
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
        Norm();
    }
    void SetBound()
    {
        if(!xs || !ys || !dat) return;
        maxval = (T)-1e11;
        minval = (T)1e11;
        avgval = 0;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                T t = dat[qp];
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
    }
    void Clean()
    {
        if(dat)
        {
            delete[] dat;
            dat = 0;
            xs = ys = 0;
        }
    }
    ~SWorkImg()
    {
        Clean();
    }
    T* operator[] (int y)
    {
        if(y >= ys) y = ys - 1;
        else if(y < 0) y = 0;
        return &dat[y * xs];
    }
    const T* operator[](int y) const
    {
        int iy = y;
        if(iy >= ys) iy = ys - 1;
        else if(iy < 0) iy = 0;
        return &dat[iy * xs];
    }
    SWorkImg& operator= (SWorkImg& tc)
    {
        if(&tc == this) return *this;
        if(xs != tc.xs || ys != tc.ys)
        {
            Clean();
            xs = tc.xs;
            ys = tc.ys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return *this;
            }
        }

        maxval = tc.maxval;
        minval = tc.minval;
        avgval = tc.avgval;
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            dat[qp] = tc[q][p];
        }
        return *this;
    }
    void operator=(SDisImg& tc)
    {
        if(xs != tc.xs || ys != tc.ys)
        {
            Clean();
            xs = tc.xs;
            ys = tc.ys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return;
            }
        }

        maxval = 0;
        minval = (T)10000;
        avgval = 0;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                int p = qp % xs;
                int q = qp / xs;
                T t;
                t = (T)(tc[q][p] & 0xff);

                dat[qp] = t;
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
        Norm();
    }
    bool GetAligned(SDisImg& tc, int align, int channel = 0)
    {
        int tcxs = tc.xs, tcys = tc.ys;
        int modxs = tcxs % align, modys = tcys % align;
        tcxs -= modxs; tcys -= modys;
        if(tcxs < align || tcys < align) return false;
        if(xs != tcxs || ys != tcys)
        {
            Clean();
            xs = tcxs;
            ys = tcys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return false;
            }
        }

        maxval = 0;
        minval = (T)10000;
        avgval = 0;
        modxs >>= 1; modys >>= 1;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                int p = qp % xs;
                int q = qp / xs;
                T t;
                if(!channel)
                    t = (T)(tc[q + modys][p + modxs] & 0xff);
                else if(channel == 1)
                    t = (T)((tc[q + modys][p + modxs] & 0xff00) >> 8);
                else
                    t = (T)((tc[q + modys][p + modxs] & 0xff0000) >> 16);
                dat[qp] = t;
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
        Norm();
        return true;
    }
    bool GetAligned8(SDisImg& tc, int align)
    {
        int tcxs = tc.xs, tcys = tc.ys;
        int modxs = tcxs % align, modys = tcys % align;
        tcxs -= modxs; tcys -= modys;
        if(tcxs < align || tcys < align) return false;
        if(xs != tcxs || ys != tcys)
        {
            Clean();
            xs = tcxs;
            ys = tcys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return false;
            }
        }

        maxval = 0;
        minval = (T)10000;
        avgval = 0;

        modxs >>= 1; modys >>= 1;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                int p = qp % xs;
                int q = qp / xs;
                T t = (T)(tc[q + modys][p + modxs]);
                dat[qp] = t;
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
        Norm();
        return true;
    }
    // kaggle
    bool GetRenormed(SDisImg& tc)
    {
        int tcxs = tc.xs, tcys = tc.ys;
        if(tcxs < HFRAME || tcys < HFRAME) return false;
        if(xs != tcxs || ys != tcys)
        {
            Clean();
            xs = tcxs;
            ys = tcys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return false;
            }
        }

        maxval = 0;
        minval = (T)10000;
        avgval = 0;
#pragma omp parallel reduction(+: avgval)
        {
            T temp_maxval = (T)0;
            T temp_minval = (T)10000;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                int p = qp % xs;
                int q = qp / xs;
                T t;
                t = (T)(tc[q][p] & 0xff);
                dat[qp] = t;
                if(temp_maxval < t) temp_maxval = t;
                if(temp_minval > t) temp_minval = t;
                avgval += t;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_maxval > maxval) maxval = temp_maxval;
                if(temp_minval < minval) minval = temp_minval;
            }
        }
        avgval /= xs * ys;
        Renorm();
        return true;
    }
    // kaggle
    void Norm()
    {
        if(maxval <= 0) return;
        T recip = ((T)1.0) / ((T)maxval);
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] *= recip;
        }
    }
    void Renorm()
    {
        T min = (T)1e11f, max = (T)-1e11f;
#pragma omp parallel
        {
            T temp_max = (T)-1e11f;
            T temp_min = (T)1e11f;
#pragma omp for
            for(int qp =0; qp < ys*xs; qp++){
                T v = dat[qp];
                if(temp_max < v) temp_max = v;
                if(temp_min > v) temp_min = v;
            }
#pragma omp critical(minval_maxval)
            {
                if(temp_max > max) max = temp_max;
                if(temp_min < min) min = temp_min;
            }
        }

        if(abs((T)(max - min)) < ((T)1e-11f)) return;
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] -= (T)min;
            dat[qp] /= (T)(max - min);
        }
        SetBound();

    }
    void Invert()
    {
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] = ((T)1.0) - dat[qp];
        }
    }
    void GetReduced2(SWorkImg& r)
    {
        int xx = xs / 2, yy = ys / 2;

        if(!xx || !yy) return;
        if(xx != r.xs || yy != r.ys)
        {
            r.Clean();
            r.dat = new T[yy * xx];
            if(!r.dat) return;
            r.xs = xx;
            r.ys = yy;
        }

        r.maxval = 0;
        r.minval = (T)10000;
        r.avgval = 0;
        for(int q = 0; q < yy; ++q)
        {
            for(int p = 0; p < xx; ++p)
            {
                int qq = 2 * q, pp = 2 * p;
                T t = dat[qq * xs + pp];
                t += dat[qq * xs + pp + 1];
                t += dat[(qq + 1) * xs + pp];
                t += dat[(qq + 1) * xs + pp + 1];
                t *= ((T)0.25);
                r.dat[q * r.xs + p] = t;
                if(r.maxval < t) r.maxval = t;
                if(r.minval > t) r.minval = t;
                r.avgval += t;
            }
        }
        r.avgval /= r.xs * r.ys;
        r.Norm();
        return;
    }
    void GetAugmented2(SWorkImg& r)
    {
        int xx = xs * 2, yy = ys * 2;

        if(!xx || !yy) return;
        if(xx != r.xs || yy != r.ys)
        {
            r.Clean();
            r.dat = new T[yy * xx];
            if(!r.dat) return;
            r.xs = xx;
            r.ys = yy;
        }

        r.maxval = 0;
        r.minval = (T)10000;
        r.avgval = 0;
        for(int q = 0; q < ys; ++q)
        {
            int qq = q * 2;
            int q1 = q + 1;
            if(q1 >= ys) q1 = ys - 1;
            for(int p = 0; p < xs; ++p)
            {
                int pp = p * 2;
                int p1 = p + 1;
                if(p1 >= xs) p1 = xs - 1;
                T t00 = dat[q * xs + p];
                T t01 = dat[q * xs + p1];
                T t10 = dat[q1 * xs + p];
                T t11 = dat[q1 * xs + p1];
                r.dat[qq * r.xs + pp] = t00;
                r.dat[qq * r.xs + pp + 1] = ((T)0.5) * (t00 + t01);
                r.dat[(qq + 1) * r.xs + pp] = ((T)0.5) * (t00 + t10);
                r.dat[(qq + 1) * r.xs + pp + 1] = ((T)0.25) * (t00 + t01 + t10 + t11);
            }
        }
        r.avgval = avgval;
        r.maxval = maxval;
        r.minval = minval;
    }
    void GetDispChannel(SDisImg& r, int channel = 0)
    {
        if(!xs || !ys) return;
        if(!channel)
        {
            if(xs != r.xs || ys != r.ys)
            {
                r.Clean();
                r.dat = new unsigned long[ys * xs];
                if(!r.dat) return;
                r.xs = xs;
                r.ys = ys;
            }
        }

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p];
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte b;
                b = (byte)t;
                if(!channel)
                    r.dat[q * xs + p] = b;
                else if(channel == 1)
                    r.dat[q * xs + p] += b << 8;
                else
                    r.dat[q * xs + p] += b << 16;
            }
        }

    }
    void GetDispImg(SDisImg& r)
    {
        if(!xs || !ys) return;
        if(xs != r.xs || ys != r.ys)
        {
            r.Clean();
            r.dat = new unsigned long[ys * xs];
            if(!r.dat) return;
            r.xs = xs;
            r.ys = ys;
        }

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p];
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte R, g, b;
                R = g = b = (byte)t;
                r.dat[q * xs + p] = (R << 16) + (g << 8) + b;
            }
        }

    }
    void GetNormalizedDispImg(SDisImg& r)
    {
        if(!xs || !ys) return;
        if(xs != r.xs || ys != r.ys)
        {
            r.Clean();
            r.dat = new unsigned long[ys * xs];
            if(!r.dat) return;
            r.xs = xs;
            r.ys = ys;
        }

        T min = (T)1e11f, max = (T)-1e11f;
        for(int q = 0; q < ys; ++q)
            for(int p = 0; p < xs; ++p)
            {
                T v = dat[q * xs + p];
                if(v < min) min = v;
                if(v > max) max = v;
            }

        if(abs((T)(max - min)) < ((T)1e-11f)) return;

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p] - min; t /= max - min;
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte R, g, b;
                R = g = b = (byte)t;
                r.dat[q * xs + p] = (R << 16) + (g << 8) + b;
            }
        }

    }
    void GetSignDispImg(SDisImg& r)
    {
        if(!xs || !ys) return;
        if(xs != r.xs || ys != r.ys)
        {
            r.Clean();
            r.dat = new unsigned long[ys * xs];
            if(!r.dat) return;
            r.xs = xs;
            r.ys = ys;
        }

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p];
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte R, G, B;
                if(t < 0) { B = -(byte)t; G = -(byte)(2 * t / 3); R = 0; }
                else { R = (byte)t; G = (byte)t / 3; B = 0; }
                r.dat[q * xs + p] = (R << 16) + (G << 8) + B;
            }
        }

    }
    void GetColorDispImg(SDisImg& r)
    {
        if(!xs || !ys) return;
        if(xs != r.xs || ys != r.ys)
        {
            r.Clean();
            r.dat = new unsigned long[ys * xs];
            if(!r.dat) return;
            r.xs = xs;
            r.ys = ys;
        }

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p];
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte R(0), G(0), B(0);
                if(t > 5)
                {
                    R = (byte)t;
                    B = 0xff - (byte)t;
                    G = 0x7f;
                }
                else if(t > 0)
                {
                    B = 0x7f;
                }
                r.dat[q * xs + p] = (R << 16) + (G << 8) + B;
            }
        }

    }
    void GetSaveImg(SSavImg& r)
    {
        if(!xs || !ys) return;
        {
            if(r.dat) delete[] r.dat;
            r.dat = new unsigned char[3 * ys * xs];
            if(!r.dat) return;
            r.xs = xs;
            r.ys = ys;
        }

        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T t = dat[q * xs + p];
                t *= 0xff;
                if(t > (T)0xff) t = (T)0xff;
                byte R, g, b;
                R = g = b = (byte)t;
                r.dat[(q * xs + p) * 3] = R;
                r.dat[(q * xs + p) * 3 + 1] = R;
                r.dat[(q * xs + p) * 3 + 2] = R;
            }
        }

    }
    SWorkImg& GetDiv(SWorkImg& u, SWorkImg& v, bool bZeroBound = true)
    {
        if(u.xs != v.xs || u.ys != v.ys)
            return *this;
        if(xs != u.xs || ys != u.ys)
        {
            Clean();
            xs = u.xs;
            ys = u.ys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return *this;
            }
        }
        #if 1
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T div;

                if(!p)
                    div = u[q][p];
                else if(p == xs - 1)
                    div = -u[q][p - 1];
                else
                    div = u[q][p] - u[q][p - 1];

                if(!q)
                    div += v[q][p];
                else if(q == ys - 1)
                    div += -v[q - 1][p];
                else
                    div += v[q][p] - v[q - 1][p];

                dat[q * xs + p] = div;
            }
        }
        #else
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T div;

                if(!p)
                    div = u[q][p];
                else if(p == xs - 1)
                    div = -u[q][p - 1];
                else
                    div = u[q][p + 1] - u[q][p];

                if(!q)
                    div += v[q][p];
                else if(q == ys - 1)
                    div += -v[q - 1][p];
                else
                    div += v[q][p + 1] - v[q][p];

                dat[q * xs + p] = div;
            }
        }
        #endif
        if(bZeroBound)
        {
            for(int q = 0; q < ys; ++q)
            {
                dat[(q + 1) * xs - 1] = dat[q * xs] = 0;
            }
            for(int p = 0; p < xs; ++p)
            {
                dat[(ys - 1) * xs + p] = dat[p] = 0;
            }
        }

        return *this;
    }
    SWorkImg& GetCDiv(SWorkImg& u, SWorkImg& v, bool bZeroBound = true)
    {
        if(u.xs != v.xs || u.ys != v.ys)
            return *this;
        if(xs != u.xs || ys != u.ys)
        {
            Clean();
            xs = u.xs;
            ys = u.ys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return *this;
            }
        }
        for(int q = 1; q < ys - 1; ++q)
        {
            for(int p = 1; p < xs - 1; ++p)
            {
                T div;
                div = u[q][p + 1] - u[q][p - 1];
                div += v[q + 1][p] - v[q - 1][p];
                dat[q * xs + p] = T(0.5) * div;
            }
        }

        if(bZeroBound)
        {
            for(int q = 0; q < ys; ++q)
            {
                dat[(q + 1) * xs - 1] = dat[q * xs] = 0;
            }
            for(int p = 0; p < xs; ++p)
            {
                dat[(ys - 1) * xs + p] = dat[p] = 0;
            }
        }
        else
        {
            for(int q = 0; q < ys; ++q)
            {
                dat[(q + 1) * xs - 1] = dat[(q + 1) * xs - 2];
                dat[q * xs] = dat[q * xs + 1];
            }
            for(int p = 0; p < xs; ++p)
            {
                dat[(ys - 1) * xs + p] = dat[(ys - 2) * xs + p];
                dat[p] = dat[xs + p];
            }
        }

        return *this;
    }
    void GetImgGrad(SWorkImg& gx, SWorkImg& gy, bool bZeroBound = true)
    {
        if(!xs || !ys) return;
        if(xs != gx.xs || ys != gx.ys)
        {
            gx.Clean();
            gx.dat = new T[ys * xs];
            if(!gx.dat) return;
            gx.xs = xs;
            gx.ys = ys;
        }
        if(xs != gy.xs || ys != gy.ys)
        {
            gy.Clean();
            gy.dat = new T[ys * xs];
            if(!gy.dat) return;
            gy.xs = xs;
            gy.ys = ys;
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            if(p >=xs - 1)
                continue;
            gx[q][p] = dat[qp + 1] - dat[qp];
        }
        if(bZeroBound)
        {
#pragma omp parallel for
            for(int q = 0; q < ys; ++q)
            {
                gx[q][xs - 1] = 0;
                gx[q][0] = 0;
            }
        }
        else
        {
#pragma omp parallel for
            for(int q = 0; q < ys; ++q)
                gx[q][xs - 1] = gx[q][xs - 2];
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            if(q >= ys - 1)
                continue;
            gy[q][p] = dat[(q + 1) * xs + p] - dat[qp];
        }
        if(bZeroBound)
        {
#pragma omp parallel for
            for(int p = 0; p < xs; ++p)
            {
                gy[ys - 1][p] = 0;
                gy[0][p] = 0;
            }
        }
        else
        {
#pragma omp parallel for
            for(int p = 0; p < xs; ++p)
                gy[ys - 1][p] = gy[ys - 2][p];
        }

    }
    void GetImgCGrad(SWorkImg& gx, SWorkImg& gy, bool bZeroBound = true)
    {
        if(!xs || !ys) return;
        if(xs != gx.xs || ys != gx.ys)
        {
            gx.Clean();
            gx.dat = new T[ys * xs];
            if(!gx.dat) return;
            gx.xs = xs;
            gx.ys = ys;
        }
        if(xs != gy.xs || ys != gy.ys)
        {
            gy.Clean();
            gy.dat = new T[ys * xs];
            if(!gy.dat) return;
            gy.xs = xs;
            gy.ys = ys;
        }

        for(int q = 1; q < ys - 1; ++q)
            for(int p = 1; p < xs - 1; ++p)
                gx[q][p] = T(0.5) * (dat[q * xs + p + 1] - dat[q * xs + p - 1]);
        for(int q = 1; q < ys - 1; ++q)
            for(int p = 1; p < xs - 1; ++p)
                gy[q][p] = T(0.5) * (dat[(q + 1) * xs + p] - dat[(q - 1) * xs + p]);

        if(bZeroBound)
        {
            for(int q = 0; q < ys; ++q)
            {
                gx[q][xs - 1] = gx[q][0] = 0;
                gy[q][xs - 1] = gy[q][0] = 0;
            }
            for(int p = 0; p < xs; ++p)
            {
                gx[ys - 1][p] = gx[0][p] = 0;
                gy[ys - 1][p] = gy[0][p] = 0;
            }
        }
        else
        {
            for(int q = 0; q < ys; ++q)
            {
                gx[q][xs - 1] = gx[q][xs - 2];
                gx[q][0] = gx[q][1];
                gy[q][xs - 1] = gy[q][xs - 2];
                gy[q][0] = gy[q][1];
            }
            for(int p = 0; p < xs; ++p)
            {
                gx[ys - 1][p] = gx[ys - 2][p];
                gx[0][p] = gx[1][p];
                gy[ys - 1][p] = gy[ys - 2][p];
                gy[0][p] = gy[1][p];
            }
        }
    }
    inline T GetInterpolated(T x, T y)
    {
        int ix = (int)x;
        int iy = (int)y;
        T mx = x - ix;
        T my = y - iy;
        if(ix < 0) ix = 0;
        if(ix > xs - 2) ix = xs - 2;
        if(iy < 0) iy = 0;
        if(iy > ys - 2) iy = ys - 2;
        T& r00 = dat[iy * xs + ix];
        T& r10 = dat[iy * xs + ix + 1];
        T& r01 = dat[(iy + 1) * xs + ix];
        T& r11 = dat[(iy + 1) * xs + ix + 1];
        T wx = ((T)1) - mx;
        T wy = ((T)1) - my;
        return r00 * wx * wy + r10 * mx * wy + r01 * my * wx + r11 * mx * my;
    }
    void Get1stDirectionalC(SWorkImg& gd, T dx, T dy)
    {
        if(!xs || !ys) return;
        if(xs != gd.xs || ys != gd.ys)
        {
            gd.Clean();
            gd.dat = new T[ys * xs];
            if(!gd.dat) return;
            gd.xs = xs;
            gd.ys = ys;
        }
        for(int q = 0; q < ys; ++q)
            for(int p = 0; p < xs; ++p)
            {
                T vp = GetInterpolated(p + dx, q + dy);
                T vm = GetInterpolated(p - dx, q - dy);
                gd[q][p] = T(0.5) * (vp - vm);
            }

        for(int q = 0; q < ys; ++q)
        {
            gd[q][xs - 1] = gd[q][0] = 0;
        }
        for(int p = 0; p < xs; ++p)
        {
            gd[ys - 1][p] = gd[0][p] = 0;
        }
    }
    void Get2ndDirectionalC(SWorkImg& gd, T dx, T dy)
    {
        if(!xs || !ys) return;
        if(xs != gd.xs || ys != gd.ys)
        {
            gd.Clean();
            gd.dat = new T[ys * xs];
            if(!gd.dat) return;
            gd.xs = xs;
            gd.ys = ys;
        }
        for(int q = 0; q < ys; ++q)
            for(int p = 0; p < xs; ++p)
            {
                T vp = GetInterpolated(p + dx, q + dy);
                T vm = GetInterpolated(p - dx, q - dy);
                gd[q][p] = vp + vm - 2 * dat[q * xs + p];
            }

        for(int q = 0; q < ys; ++q)
        {
            gd[q][xs - 1] = gd[q][0] = 0;
        }
        for(int p = 0; p < xs; ++p)
        {
            gd[ys - 1][p] = gd[0][p] = 0;
        }
    }
    void GetImgHesse(SWorkImg& gxx, SWorkImg& gxy, SWorkImg& gyy)
    {
        if(!xs || !ys) return;
        if(xs != gxx.xs || ys != gxx.ys)
        {
            gxx.Clean();
            gxx.dat = new T[ys * xs];
            if(!gxx.dat) return;
            gxx.xs = xs;
            gxx.ys = ys;
        }
        if(xs != gxy.xs || ys != gxy.ys)
        {
            gxy.Clean();
            gxy.dat = new T[ys * xs];
            if(!gxy.dat) return;
            gxy.xs = xs;
            gxy.ys = ys;
        }
        if(xs != gyy.xs || ys != gyy.ys)
        {
            gyy.Clean();
            gyy.dat = new T[ys * xs];
            if(!gyy.dat) return;
            gyy.xs = xs;
            gyy.ys = ys;
        }
        for(int q = 1; q < ys - 1; ++q)
            for(int p = 1; p < xs - 1; ++p)
                gxx[q][p] = (dat[q * xs + p + 1] + dat[q * xs + p - 1] - 2 * dat[q * xs + p]);
        for(int q = 1; q < ys - 1; ++q)
            for(int p = 1; p < xs - 1; ++p)
                gxy[q][p] = T(0.25) * (dat[(q + 1) * xs + p + 1] + dat[(q - 1) * xs + p - 1] - dat[(q + 1) * xs + p - 1] - dat[(q - 1) * xs + p + 1]);
        for(int q = 1; q < ys - 1; ++q)
            for(int p = 1; p < xs - 1; ++p)
                gyy[q][p] = (dat[(q + 1) * xs + p] + dat[(q - 1) * xs + p] - 2 * dat[q * xs + p]);

        for(int q = 0; q < ys; ++q)
        {
            gxx[q][xs - 1] = gxx[q][0] = 0;
            gxy[q][xs - 1] = gxy[q][0] = 0;
            gyy[q][xs - 1] = gyy[q][0] = 0;
        }
        for(int p = 0; p < xs; ++p)
        {
            gxx[ys - 1][p] = gxx[0][p] = 0;
            gxy[ys - 1][p] = gxy[0][p] = 0;
            gyy[ys - 1][p] = gyy[0][p] = 0;
        }
    }
    SWorkImg& GetMagnitude(SWorkImg& u, SWorkImg& v, T beta = T(0))
    {
        if(u.xs != v.xs || u.ys != v.ys)
            return *this;
        if(xs != u.xs || ys != u.ys)
        {
            Clean();
            xs = u.xs;
            ys = u.ys;
            dat = new T[ys * xs];
            if(!dat)
            {
                xs = ys = 0;
                return *this;
            }
        }
        for(int q = 0; q < ys; ++q)
        {
            for(int p = 0; p < xs; ++p)
            {
                T mag = u[q][p] * u[q][p] + v[q][p] * v[q][p];

                dat[q * xs + p] = sqrt(mag + beta);
            }
        }

        return *this;
    }
    void GetLaplace(SWorkImg& s)
    {
        if(xs != s.xs || ys != s.ys)
            return;
        for(int yy = 1; yy < ys - 1; ++yy)
        {
            for(int xx = 1; xx < xs - 1; ++xx)
            {
                dat[yy * xs + xx] = -4 * s[yy][xx] + s[yy][xx + 1] + s[yy][xx - 1] + s[yy + 1][xx] + s[yy - 1][xx];
            }
        }
        int y = ys - 1;
        for(int xx = 1; xx < xs - 1; ++xx)
        {
            dat[xx] = -4 * s[0][xx] + s[0][xx + 1] + s[0][xx - 1] + s[1][xx] + s[1][xx];
            dat[y * xs + xx] = -4 * s[y][xx] + s[y][xx + 1] + s[y][xx - 1] + s[y - 1][xx] + s[y - 1][xx];
        }
        int x = xs - 1;
        for(int yy = 1; yy < ys - 1; ++yy)
        {
            dat[yy * xs + 0] = -4 * s[yy][0] + s[yy + 1][0] + s[yy - 1][0] + s[yy][1] + s[yy][1];
            dat[yy * xs + x] = -4 * s[yy][x] + s[yy + 1][x] + s[yy - 1][x] + s[yy][x - 1] + s[yy][x - 1];
        }
        dat[0 * xs + 0] = 0.5f * (dat[0 * xs + 1] + dat[1 * xs + 0]);
        dat[0 * xs + x] = 0.5f * (dat[0 * xs + x - 1] + dat[1 * xs + x]);
        dat[y * xs + 0] = 0.5f * (dat[(y - 1) * xs + 0] + dat[y * xs + 1]);
        dat[y * xs + x] = 0.5f * (dat[(y - 1) * xs + x] + dat[y * xs + x - 1]);

    }

    int GetWidth() { return xs;}
    int GetHeight() {return ys;}

    SWorkImg& operator= (T r)
    {
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] = r;
        }
        return *this;
    }
    SWorkImg& operator*= (T r)
    {
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] *= r;
        }
        return *this;
    }
    SWorkImg& operator+= (T r)
    {
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] += r;
        }
        return *this;
    }
    SWorkImg& operator-= (T r)
    {
        #pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            dat[qp] -= r;
        }
        return *this;
    }
    SWorkImg& operator+= (SWorkImg& tc)
    {
        if(xs != tc.xs || ys != tc.ys)
        {
            return *this;
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            dat[qp] += tc[q][p];
        }
        return *this;
    }
    SWorkImg& operator-= (SWorkImg& tc)
    {
        if(xs != tc.xs || ys != tc.ys)
        {
            return *this;
        }
#pragma omp parallel for
        for(int qp =0; qp < ys*xs; qp++){
            int p = qp % xs;
            int q = qp / xs;
            dat[qp] -= tc[q][p];
        }
        return *this;
    }

    T* dat;
    T maxval;
    T minval;
    T avgval;
    int xs;
    int ys;
};

