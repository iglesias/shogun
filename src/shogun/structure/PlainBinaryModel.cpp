/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando José Iglesias García
 * Copyright (C) 2013 Fernando José Iglesias García
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/structure/PlainBinaryModel.h>
#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

CPlainBinaryModel::CPlainBinaryModel()
: CStructuredModel()
{
}

CPlainBinaryModel::CPlainBinaryModel(CFeatures* features, CStructuredLabels* labels)
: CStructuredModel(features, labels)
{
}

CPlainBinaryModel::~CPlainBinaryModel()
{
}

const char* CPlainBinaryModel::get_name() const
{
	return "PlainBinaryModel";
}

int32_t CPlainBinaryModel::get_dim() const
{
	return 2;
}

SGVector<float64_t> CPlainBinaryModel::get_joint_feature_vector(int32_t feat_idx, CStructuredData* y)
{
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	SGVector<float64_t> x = 
		((CDotFeatures*) m_features)->get_computed_dot_feature_vector(feat_idx);
	REQUIRE(x.vlen == 1, "PlainBinaryModel's features must be unidimensional\n");

	CRealNumber* label = CRealNumber::obtain_from_generic(y);
	ASSERT(label != NULL);
	float64_t lvalue = label->value;
	REQUIRE(lvalue == +1.0 || lvalue == -1.0, "PlainBinaryModel's labels must"
			" be equal to +1 or -1\n");

	if (lvalue == +1.0)
		psi[0] = x[0];
	else if (lvalue == -1.0)
		psi[1] = x[0];
	else
		SG_ERROR("Cannot happen\n");

	return psi;
}

CResultSet* CPlainBinaryModel::argmax(SGVector< float64_t > w, int32_t feat_idx, bool const training)
{
	ASSERT(get_dim() == w.vlen);
	CResultSet* ret = new CResultSet();

	// Find the class (negative or positive) that gives the maximum score

	CRealNumber POS_LABEL = CRealNumber(+1);
	CRealNumber NEG_LABEL = CRealNumber(-1);

	SGVector<float64_t> pos_psi = get_joint_feature_vector(feat_idx, &POS_LABEL);
	SGVector<float64_t> neg_psi = get_joint_feature_vector(feat_idx, &NEG_LABEL);

	float64_t pos_score = SGVector<float64_t>::dot(pos_psi.vector, w.vector, w.vlen);
	float64_t neg_score = SGVector<float64_t>::dot(neg_psi.vector, w.vector, w.vlen);

	if (training)
	{
		// Loss-augmented prediction
		pos_score += CStructuredModel::delta_loss(feat_idx, &POS_LABEL);
		neg_score += CStructuredModel::delta_loss(feat_idx, &NEG_LABEL);

		ret->psi_truth = CStructuredModel::get_joint_feature_vector(feat_idx, feat_idx);
		ret->score = -SGVector<float64_t>::dot(w.vector, ret->psi_truth.vector, w.vlen);
	}

	if (pos_score >= neg_score)		// positive class predicted
	{
		ret->psi_pred	= pos_psi;
		ret->score		+= pos_score;
		ret->argmax		= new CRealNumber(POS_LABEL.value);

		if (training)
			ret->delta	= CStructuredModel::delta_loss(feat_idx, &POS_LABEL);
	}
	else							// negative class predicted
	{
		ret->psi_pred	= neg_psi;
		ret->score		+= neg_score;
		ret->argmax		= new CRealNumber(NEG_LABEL.value);

		if (training)
			ret->delta	= CStructuredModel::delta_loss(feat_idx, &NEG_LABEL);
	}

	return ret;
}

float64_t CPlainBinaryModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	CRealNumber* label1 = CRealNumber::obtain_from_generic(y1);
	CRealNumber* label2 = CRealNumber::obtain_from_generic(y2);
	ASSERT(label1 != NULL && label2 != NULL);

	return label1->value != label2->value;
}
