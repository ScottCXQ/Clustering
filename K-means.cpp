#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <assert.h>
#include "Kmeans.h";

using namespace std;

KMeans::KMeans(int dimNum, int clusterNum)
{
    m_dimNum = dimNum;
    m_clusterNum = clusterNum;

    m_means = new double*[m_clusterNum];
    for (int i = 0; i < m_clusterNum; i++)
    {
        m_means[i] = new double[m_dimNum];
        memset(m_means[i], 0, sizeof(double) * m_dimNum);
    }

    m_initMode = InitRandom;
    m_maxIterNum = 100;
    m_endError = 0.001;
}

KMeans::~KMeans()
{
    for (int i = 0; i < m_clusterNum; i++)
    {
        delete[] m_means[i];
    }
    delete[] m_means;
}

void KMeans::Cluster(const char *sampleFileName, const char *labelFileName)
{
    ifstream sampleFile(sampleFileName, ios_base::binary);
    assert(sampleFile);

    int size = 0;
    int dim = 0;
    sampleFile.read((char*)&size, sizeof(int));
    sampleFile.read((char*)&dim, sizeof(int));
    assert(size >= m_clusterNum);
    assert(dim == m_dimNum);

    Init(sampleFile);


    //递归
    double* x = new double[m_dimNum];
    int label = -1;
    double iterNum = 0;
    double lastCost = 0;
    double currentCost = 0;
    int unchanged = 0;
    bool loop = true;
    int* counts = new int[m_clusterNum];
    double** next_means = new double*[m_clusterNum];
    for (int i = 0; i < m_clusterNum; i++)
    {
       next_means[i] = new double[m_dimNum];
    }

    while(loop)
    {
        memset(counts, 0, sizeof(int)*m_clusterNum);
        for (int i = 0; i < m_clusterNum; i++)
        {
            memset(next_means[i], 0, sizeof(double)*m_dimNum);
        }
        lastCost = currentCost;
        currentCost = 0;

        sampleFile.clear();
        sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

        for (int i = 0; i < size; i++)
        {
            sampleFile.read((char*)x, sizeof(double) * m_dimNum);
            currentCost += GetLabel(x, &label);

            counts[label]++;
            for (int d = 0; d < m_dimNum; d++)
            {
                next_means[label][d] += x[d];
            }
        }
        currentCost /= size;

        for (int i = 0; i < m_clusterNum; i++)
        {
            if (counts[i] > 0)
            {
                for (int d=0;d<m_dimNum;d++)
                {
                    next_means[i][d] /= counts[i];
                }
                memcpy(m_means[i], next_means[i], sizeof(double) * m_dimNum);
            }
        }

        iterNum++;
        if(fabs(lastCost-currentCost) < m_endError * lastCost)
        {
            unchanged++;
        }
        if(iterNum >= m_maxIterNum || unchanged >= 3)
        {
            loop=false;
        }

        ofstream labelFile(labelFileName, ios_base::binary);
        assert(labelFile);

        labelFile.write((char*)&size, sizeof(int));
        sampleFile.clear();
        sampleFile.seekg(sizeof(int) * 2, ios_base::beg);

        for (int i = 0; i < size; i++)
        {
            sampleFile.read((char*)x, sizeof(double) * m_dimNum);
            GetLabel(x, &label);
            labelFile.write((char*)&label, sizeof(int));
        }
        sampleFile.close();
        labelFile.close();
        delete[] counts;
        delete[] x;
        for (int i = 0; i < m_clusterNum; i++)
        {
            delete[] next_means;
        }
        delete[] next_means;
    }
}

//N为特征向量数目
void KMeans::Cluster(double *data, int N, int *Label)
{
    int size = 0;
    size = N;
    assert(size >= m_clusterNum);

    Init(data, N);
    double* x = new double[m_dimNum];
    int label = -1;
    double iterNum = 0;
    double lastCost = 0;
    double currentCost = 0;
    int unchanged = 0;
    bool loop = true;
    int* counts = new int[m_clusterNum];
    double** next_means = new double*[m_clusterNum];

    for(int i = 0; i < m_clusterNum; i++)
    {
        next_means[i] = new double[m_dimNum];
    }
    while(loop)
    {
        memset(counts, 0, sizeof(int) * m_clusterNum);
        for(int i = 0; i < m_clusterNum; i++)
        {
            memset(next_means[i], 0, sizeof(double) * m_dimNum);
        }

        lastCost = currentCost;
        currentCost = 0;

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < m_dimNum; j++)
            {
                x[j] = data[i*m_dimNum + j];
            }
            currentCost += GetLabel(x, &label);

            counts[label]++;
            for (int d=0;d<m_dimNum;d++)
            {
                next_means[label][d] += x[d];
            }
        }
        currentCost /= size;

        for (int i=0;i<m_clusterNum;i++)
        {
            if(counts[i]>0)
            {
                for (int d=0;d<m_dimNum;d++)
                {
                    next_means[i][d] /= counts[i];
                }
                memcpy(m_means[i],next_means[i],sizeof(double)*m_dimNum);
            }
        }

        iterNum++;
        if(fabs(lastCost - currentCost) < m_endError * lastCost)
        {
            unchanged++;
        }
        if(iterNum >= m_maxIterNum || unchanged >= 3)
        {
            loop = false;
        }
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < m_dimNum; j++)
        {
            x[j] = data[i*m_dimNum + j];
        }
        GetLabel(x, &label);
        Label[i] = label;
    }

    delete[] counts;
    delete[] x;
    for (int i = 0; i < m_clusterNum; i++)
    {
        delete[] next_means[i];
    }
    delete[] next_means;
}

void KMeans::Init(double *data, int N)
{
    int size = N;
    if(m_initMode == InitRandom)
    {

        int interval = size / m_clusterNum;
        double* sample = new double[m_dimNum];

        srand((unsigned)time(NULL));

        for (int i = 0; i < m_clusterNum; i++)
        {
            int select = interval * i + (interval-1)*rand() / RAND_MAX;
            for(int j=0;j<m_dimNum;j++)
            {
                sample[j] = data[select*m_dimNum + j];
                memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
            }
            delete[] sample;
        }
    }
        else if(m_initMode == InitUniform)
        {
            double* sample = new double[m_dimNum];

            for(int i = 0; i < m_clusterNum; i++)
            {
                int select = i*size / m_clusterNum;
                for (int j = 0; j < m_dimNum; j++)
                {
                    sample[j] = data[select*m_dimNum + j];
                }
                memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
            }
            delete[] sample;
        }
        else if (m_initMode == InitManual)
        {
            //Do Nothing
        }
}

void KMeans::Init(ifstream& sampleFile)
{
    int size = 0;
    sampleFile.seekg(0, ios_base::beg);
    sampleFile.read((char*)&size,sizeof(int));

    if(m_initMode == InitRandom)
    {
        int interval = size / m_clusterNum;
        double* sample = new double[m_dimNum];

        srand((unsigned)time(NULL));

        for(int i=0;i<m_clusterNum;i++)
        {
            int select = interval * i + (interval - 1) * rand() /RAND_MAX;
            int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;

            sampleFile.seekg(offset, ios_base::beg);
            sampleFile.read((char*)&sample, sizeof(double) * m_dimNum);
            memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
        }
        delete[] sample;
    }
    else if(m_initMode == InitUniform)
    {
        double* sample = new double[m_dimNum];
        for (int i=0;i<m_clusterNum;i++)
        {
            int select = i * size / m_clusterNum;
            int offset = sizeof(int) * 2 + select * sizeof(double) * m_dimNum;
            sampleFile.seekg(offset, ios_base::beg);
            sampleFile.read((char*)&sample, sizeof(double) * m_dimNum);
            memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
        }
        delete[] sample;
    }
    else if(m_initMode == InitManual)
    {
        //Do Nothing
    }
}

double KMeans::GetLabel(const double *sample, int *label)
{
    double dist = -1;
    for (int i=0;i<m_clusterNum;i++)
    {
        double temp = CalcDistance(sample, m_means[i],m_dimNum);
        if(temp < dist || dist == -1)
        {
            dist = temp;
            *label = i;
        }
    }
    return dist;
}

double KMeans::CalcDistance(const double *x, const double *y, int dimNum)
{
    double temp = 0;
    for(int d=0;d<dimNum;d++)
    {
        temp += (x[d] - y[d]) * (x[d] - y[d]);
    }

    return sqrt(temp);
}

ostream &operator<<(ostream &out, KMeans &kmeans)
{
    out<<"<KMeans>"<<endl;
    out<<"<DimNum>"<<kmeans.m_dimNum<<"</DimNum>"<<endl;

    out<<"<Mean>"<<endl;
    for(int i=0;i<kmeans.m_clusterNum;i++)
    {
        for(int d=0;d<kmeans.m_dimNum;d++)
        {
            out<<kmeans.m_means[i][d]<<" ";
        }
        out<<endl;
    }
    out<<"</Mean>"<<endl;

    out<<"</KMeans>"<<endl;

    return out;
}
