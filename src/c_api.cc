// Copyright (c) 2016 by Contributors
#include <tinyflow/base.h>
#include <tinyflow/c_api.h>

/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int NNAPIHandleException(const dmlc::Error &e) {
  NNAPISetLastError(e.what());
  return -1;
}

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return NNAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &_except_) { Finalize; return NNAPIHandleException(_except_); } return 0; // NOLINT(*)

/*! \brief entry to to easily hold returning information */
struct TinyAPIThreadLocalEntry {
  /*! \brief result holder for returning handles */
  std::vector<const float*> floatp;
  /*! \brief result holder for returning handles */
  std::vector<nn_uint> dtype;
  /*! \brief result holder for returning handles */
  std::vector<nn_uint> shape_ndim;
  /*! \brief result holder for returning handles */
  std::vector<const nn_uint*> shape_data;
};

using namespace tinyflow;

int NNSessionCreate(SessionHandle* handle, const char* option) {
  API_BEGIN();
  *handle = Session::Create(option);
  API_END();
}

int NNSessionClose(SessionHandle handle) {
  API_BEGIN();
  delete static_cast<Session*>(handle);
  API_END();
}

int NNSessionRun(SessionHandle handle,
                 SymbolHandle graph,
                 nn_uint num_feed,
                 const SymbolHandle* feed_placeholders,
                 const float** feed_dptr,
                 const nn_uint* feed_dtype,
                 const nn_uint* feed_shape_csr_ptr,
                 const nn_uint* feed_shape_data,
                 nn_uint* num_out,
                 const float*** out_dptr,
                 const nn_uint** out_dtype,
                 const nn_uint** out_shape_ndim,
                 const nn_uint*** out_shape_data) {
  API_BEGIN();
  std::unordered_map<std::string, TBlob> feed;
  for (nn_uint i = 0; i < num_feed; ++i) {
    const std::string& key =
        static_cast<nnvm::Symbol*>(feed_placeholders[i])->outputs[0].node->attrs.name;
    TBlob tmp;
    tmp.data = (void*)feed_dptr[i];  // NOLINT(*)
    tmp.shape = TShape(feed_shape_data + feed_shape_csr_ptr[i],
                       feed_shape_data + feed_shape_csr_ptr[i + 1]);
    feed[key] = tmp;
  }

  const std::vector<TBlob>& out = static_cast<Session*>(handle)->Run(
      static_cast<nnvm::Symbol*>(graph), feed);
  *num_out = static_cast<nn_uint>(out.size());
  auto* ret = dmlc::ThreadLocalStore<TinyAPIThreadLocalEntry>::Get();
  ret->floatp.resize(out.size());
  ret->dtype.resize(out.size());
  ret->shape_ndim.resize(out.size());
  ret->shape_data.resize(out.size());

  for (size_t i = 0; i < out.size(); ++i) {
    ret->floatp[i] = static_cast<const float*>(out[i].data);
    ret->dtype[i] = out[i].dtype;
    ret->shape_ndim[i] = out[i].shape.ndim();
    ret->shape_data[i] = out[i].shape.data();
  }
  *out_dptr = dmlc::BeginPtr(ret->floatp);
  *out_dtype = dmlc::BeginPtr(ret->dtype);
  *out_shape_ndim = dmlc::BeginPtr(ret->shape_ndim);
  *out_shape_data = dmlc::BeginPtr(ret->shape_data);
  API_END();
  return 0;
}
