#include <vector>

#include <Python.h>
#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>

#include "imgwarp_mls.h"
#include "imgwarp_piecewiseaffine.h"
#include "imgwarp_mls_rigid.h"
#include "imgwarp_mls_similarity.h"

void utils_convert_points(PyArrayObject* ppoints, std::vector<cv::Point>& vpoints)
{
	int dim = PyArray_NDIM(ppoints);
	assert(dim == 2); // N*2 points array
	npy_intp* shape = PyArray_DIMS(ppoints);

	int num = static_cast<int>(shape[0]);
	int length = static_cast<int>(shape[1]);
	assert(length == 2); // [x, y]
	int* data_ptr = static_cast<int*>(PyArray_DATA(ppoints));
	for (int i = 0; i < num; i++)
	{
		int x = *data_ptr++;
		int y = *data_ptr++;
		vpoints.push_back(cv::Point(x, y));
	}
}

static PyObject* cmls_warp_func(PyObject *self, PyObject *args, PyObject* kw)
{
	PyArrayObject* src_image;
	PyArrayObject* src_points;
	PyArrayObject* dst_points;

	char* method = "Similarity";
	double alpha = 1.0;
	int grid_size = 5;

	char* keywords[] = { "image", "src_points", "dst_points", "method", "alpha", "grid_size", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kw, "O!O!O!|sdi", keywords,
		&PyArray_Type, &src_image, &PyArray_Type, &src_points, &PyArray_Type, &dst_points,
		&method, &alpha, &grid_size))
	{
		std::cerr << "Invaid input parameters." << std::endl;
		return NULL;
	}

	// decode numpy array to cv::Mat
	npy_intp* shape = PyArray_DIMS(src_image);
	const int height = static_cast<int>(shape[0]);
	const int width = static_cast<int>(shape[1]);
	const int channels = static_cast<int>(shape[2]);
	unsigned char* image_ptr = (unsigned char*)(PyArray_DATA(src_image));
	int image_type = channels == 3 ? CV_8UC3 : CV_8UC1;
	cv::Mat image(height, width, image_type, image_ptr);
	cv::Mat image2 = image.clone();

	// decode numpy src_points to points1 vector
	std::vector<cv::Point> points1;
	utils_convert_points(src_points, points1);

	// decode numpy dst_points to points2 vector
	std::vector<cv::Point> points2;
	utils_convert_points(dst_points, points2);

	// mls warp image
	ImgWarp_MLS* im_wapper = NULL;

	if (strcmp(method, "Similarity") >= 0)
	{
		im_wapper = new ImgWarp_MLS_Similarity();
	}
	else if (strcmp(method, "Rigid") >= 0)
	{
		im_wapper = new ImgWarp_MLS_Rigid();
	}
	else if (strcmp(method, "Piecewise") >= 0)
	{
		im_wapper = new ImgWarp_PieceWiseAffine();
	}
	else
	{
		std::cerr << "[ERROR] undefined method " << method << "." << std::endl;
		Py_RETURN_NONE;
	}

	cv::Mat warp_image = im_wapper->setAllAndGenerate(image2, points1, points2, width, height);
	if (im_wapper)
	{
		delete im_wapper;
		im_wapper = NULL;
	}

	npy_intp warp_dims[] = { height, width, channels };
        size_t mem_size = height * width * channels * sizeof(uchar);
        void* mem_buffer = malloc(mem_size);
        memcpy(mem_buffer, warp_image.data, mem_size);
	PyArrayObject* dst_image = (PyArrayObject*)PyArray_SimpleNewFromData(3, warp_dims, NPY_UINT8, mem_buffer);
        // set dst_image own the mem_buffer 
        dst_image->flags |= NPY_ARRAY_OWNDATA;

	return (PyObject*)dst_image;
}

static PyMethodDef cmlswarp_methods[] = {
	{
		"warp",
		(PyCFunction)cmls_warp_func,
		METH_VARARGS | METH_KEYWORDS,
		"Python implement for Moving-Least-Square image warping."
	},
	{ NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION==3
// python 3
static struct PyModuleDef cmlswarp_module = {
	PyModuleDef_HEAD_INIT,
	"cmlswarp",
	NULL,
	-1,
	cmlswarp_methods
};

PyMODINIT_FUNC PyInit_cmlswarp()
{
	import_array();
	return PyModule_Create(&cmlswarp_module);
}
#else
// python 2
PyMODINIT_FUNC initcmlswarp()
{
	import_array();
	(void)Py_InitModule("cmlswarp", cmlswarp_methods);
}
#endif


