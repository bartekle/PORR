from scipy.optimize import rosen
import numpy as np


def testFcn1(x):
    """
    testFcn1
    """

    s1 = np.sum([(2 * (i + 1) - 1) * np.power(x[i] - (2 + (i + 1)), 4)
                 for i in range(len(x))], axis=0)
    s2 = np.sum([np.power(x[i] - x[j] + (j - i), 4) for i in range(len(x) - 1)
                 for j in range(i + 1, len(x))], axis=0)

    return s1 + s2


def testFcn2(x):
    """
    testFcn2
    """

    return rosen(x)

double localmin(double a, double b, double t)
{
	double x, d, e, m, p, q, r, tol, t2, u, v, w, fu, fv, fw, fx;
	double eps = sqrt(DBL_EPSILON);
	const double c = (3 - sqrt(5)) / 2;
	v = w = x = a + c*(b - a);
	e = 0;
	fv = fw = fx = f(x);
	//main loop
	while (true)
	{
		m = 0.5*(a + b);
		tol = eps*fabs(x) + t;
		t2 = 2 * tol;
		//Check stopping criterion
		if (fabs(x - m) <= t2 - 0.5*(b - a)) break;
		p = q = r = 0;
		if (fabs(e) > tol)
		{//fit parabola
			r = (x - w)*(fx - fv);
			q = (x - v)*(fx - fw);
			p = (x - v)*q - (x - w)*r;
			q = 2 * (q - r);
			if (q > 0)p = -p; else q = -q;
			r = e; e = d;
		}
		if (fabs(p) < fabs(0.5*q*r) && p > q*(a - x) && p < q*(b - x))
		{
			// a "parabolic interpolation" step
			d = p / q;
			u = x + d;
			// f must not be evaluated too close to a or b
			if (u - a < t2 || b - u < t2)
			{
				d = x < m ? tol : -tol;
			}
		}
		else
		{  // a "golden section" step
			e = (x < m ? b : a) - x;
			d = c*e;
		}
		// f must not be evaluated too close to x
		u = x + (fabs(d) >= tol ? d : (d > 0 ? tol : -tol));
		fu = f(u);
		// update a,b,v,w and x
		if (fu <= fx)
		{
			if (u < x)b = x; else a = x;
			v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
		}
		else
		{
			if (u < x) a = u; else b = u;
			if (fu<=fw || w==x)
			{
				v = w; fv = fw; w = u; fw = fu;
			}
			else if (fu <= fv || v == x || v == w)
			{
				v = u; fv = fu;
			}
		}
	}
	return x;
}

int main()
{
	double res = localmin(0, 10, 1e-16);
	printf("min in %.16f\n", res);
	return 0;
}
