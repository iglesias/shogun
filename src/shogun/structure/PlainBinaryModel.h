/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando José Iglesias García
 * Copyright (C) 2013 Fernando José Iglesias García
 */

#ifndef _PLAIN_BINARY_MODEL__H__
#define _PLAIN_BINARY_MODEL__H__

#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/**
 * TODO DOC
 * @brief Class CPlainBinaryModel to make simple structured learning tests
 */
class CPlainBinaryModel : public CStructuredModel
{

	public:
		/** default constructor */
		CPlainBinaryModel();

		/** constructor
		 *
		 * @param features the feature vectors
		 * @param labels structured labels
		 */
		CPlainBinaryModel(CFeatures* features, CStructuredLabels* labels);

		/** destructor */
		virtual ~CPlainBinaryModel();

		/** @return name of SGSerializable */
		virtual const char* get_name() const;

		/**
		 * return the dimensionality of the joint feature space, i.e. 
		 * the dimension of the weight vector \f$w\f$
		 */
		virtual int32_t get_dim() const;

		/**
		 * get joint feature vector
		 *
		 * \f[
		 * \vec{\Psi}(\bf{x}_\text{feat\_idx}, \bf{y})
		 * \f]
		 *
		 * @param feat_idx index of the feature vector to use
		 * @param y structured label to use
		 *
		 * @return the joint feature vector
		 */
		virtual SGVector< float64_t > get_joint_feature_vector(int32_t feat_idx, CStructuredData* y);

		/**
		 * obtains the argmax of \f$ \Delta(y_{pred}, y_{truth}) +
		 * \langle w, \Psi(x_{truth}, y_{pred}) \rangle \f$
		 *
		 * @param w weight vector
		 * @param feat_idx index of the feature to compute the argmax
		 * @param training true if argmax is called during training.
		 * Then, it is assumed that the label indexed by feat_idx in
		 * m_labels corresponds to the true label of the corresponding
		 * feature vector.
		 *
		 * @return structure with the predicted output
		 */
		virtual CResultSet* argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training = true);

		/** computes \f$ \Delta(y_{1}, y_{2}) \f$
		 *
		 * @param y1 an instance of structured data
		 * @param y2 another instance of structured data
		 *
		 * @return loss value
		 */
		virtual float64_t delta_loss(CStructuredData* y1, CStructuredData* y2);

}; /* PlainBinaryModel */

} /* namespace shogun */

#endif /* _PLAIN_BINARY_MODEL__H__ */
