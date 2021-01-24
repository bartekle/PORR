#!/usr/bin/env python

import numpy as np
import pyopencl as cl


def chazanMiranker(f, x0, N=int(1e3), eps=1e-6, solutionHistory=False, beta=1., q=0.9, threads=1):
    """
    Parallel Gauss Seidel Method variant
    """
    start_time = perf_counter()
    P = x0
    fn = f(P)
    n = len(P)

    if solutionHistory:
        xHist = P.copy()
        funHist = np.array([fn])

    initialDirections = np.eye(n)
    actualDirections = initialDirections.copy()

    for it in range(N):
        xPoints = np.tile(P, (n, 1)) + \
            np.cumsum(actualDirections, axis=0)

        alphas = np.array((n,),dtype='float32')

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        mf = cl.mem_flags

        directions_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=actualDirections[0])
        xpoints_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xPoints)
        size_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=n)
        # This loop is parallelizable

        prg = cl.Program(ctx, """
        double f (double alpha, double xPoint[], double direction[], int length){

            double arg[length];

            for (int i=0; i < length; i++) {
                    arg[i] = xPoint[i] + alpha * direction[i];

            }

            return testFunc1(arg, length);
        }

        double testFunc1(double x[], int size){
            double sum1 = 0;
            double sum2 = 0;

            for (int i = 0; i <= size - 1; i++){
                sum1 += (2*(i+1) - 1) * pow((x[i] - (2 + (i+1))), 4);
            }

            for (int j = 0; j <= size - 2; j++){
                for (int z = j + 1; z <= size - 1; z++){
                    sum2 += pow((x[j] - x[z] + ((z+1) - (j+1))), 4);
                }
            }

            return sum1 + sum2;
        }
        __kernel void brent(
            __global const float *directions_g, __global const float *xpoints_g, __global const int *size_g, __global float *alphas_g)
        {
          int gid = get_global_id(0);
          double x, d, e, m, p, q, r, tol, t2, u, v, w, fu, fv, fw, fx;
            double eps = sqrt(2.220446049250313E-016);
            const double c = (3 - sqrt(5)) / 2;
            v = w = x = a + c*(b - a);
            e = 0;
            fv = fw = fx = f(x, xpoints_g[gid], directions_g, size_g);
            while (true)
            {
            	m = 0.5*(a + b);
            	tol = eps*fabs(x) + t;
            	t2 = 2 * tol;
            	if (fabs(x - m) <= t2 - 0.5*(b - a)) break;
            	p = q = r = 0;
            	if (fabs(e) > tol)
            	{
            		r = (x - w)*(fx - fv);
            		q = (x - v)*(fx - fw);
            		p = (x - v)*q - (x - w)*r;
            		q = 2 * (q - r);
            		if (q > 0)p = -p; else q = -q;
            		r = e; e = d;
            	}
            	if (fabs(p) < fabs(0.5*q*r) && p > q*(a - x) && p < q*(b - x))
            	{
            		d = p / q;
            		u = x + d;
            		if (u - a < t2 || b - u < t2)
            		{
            			d = x < m ? tol : -tol;
            		}
            	}
            	else
            	{
            		e = (x < m ? b : a) - x;
            		d = c*e;
            	}
            	u = x + (fabs(d) >= tol ? d : (d > 0 ? tol : -tol));
            	fu = f(u, xpoints_g[gid], directions_g, size_g);
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
            alphas_g[gid] = x
            }
        """).build()
        alphas_g = cl.Buffer(ctx, mf.WRITE_ONLY, xPoints[0].nbytes)
        knl = prg.sum
        knl(queue, xPoints[0].shape, None, xpoints_g, directions_g, size_g, alphas_g)
        cl.enqueue_copy(queue, alphas, alphas_g)

        Pnew = xPoints[0] + alphas[0] * actualDirections[0]
        # debug
        # print(f"Chazan: {f(Pnew)}, it: {it+1}")
        if f(Pnew) - 0 < eps:
            P = Pnew
            break
        else:
            if f(Pnew) < f(P):
                P = Pnew

        if solutionHistory:
            xHist = np.vstack((xHist, P))
            funHist = np.vstack((funHist, f(P)))

        for dirInd in range(n - 1):
            actualDirections[dirInd] = actualDirections[dirInd + 1] + \
                (alphas[dirInd + 1] - alphas[dirInd]) * \
                actualDirections[0]

        newDir = beta * initialDirections[it % initialDirections.shape[-1]]
        beta *= q

        actualDirections[-1] = newDir
    end_time = perf_counter()
    retval = {
        'x': P,
        'fun': f(P),
        'nit': it + 1,
        'time': end_time - start_time
    }

    if solutionHistory:
        retval['xHist'] = xHist
        retval['funHist'] = funHist

    return retval
