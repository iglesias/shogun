/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/StructuredModel.h>

using namespace shogun;

CStructuredModel::CStructuredModel() : CSGObject()
{
	init();
}

CStructuredModel::CStructuredModel(
		CFeatures*         features,
		CStructuredLabels* labels)
: CSGObject()
{
	init();

	m_features = features;
	m_labels   = labels;

	SG_REF(features);
	SG_REF(labels);
}

CStructuredModel::~CStructuredModel()
{
	SG_UNREF(m_labels);
	SG_UNREF(m_features);
}

void CStructuredModel::init_opt(
		SGMatrix< float64_t > A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > b,
		SGVector< float64_t > lb,
		SGVector< float64_t > ub,
		SGMatrix< float64_t > & C)
{
	SG_ERROR("init_opt is not implemented for %s!\n", get_name());
}

void CStructuredModel::set_labels(CStructuredLabels* labels)
{
	SG_UNREF(m_labels);
	SG_REF(labels);
	m_labels = labels;
}

void CStructuredModel::set_features(CFeatures* features)
{
	SG_UNREF(m_features);
	SG_REF(features);
	m_features = features;
}

SGVector< float64_t > CStructuredModel::get_joint_feature_vector(
		int32_t feat_idx, 
		int32_t lab_idx)
{
	return get_joint_feature_vector(feat_idx, m_labels->get_label(lab_idx));
}

SGVector< float64_t > CStructuredModel::get_joint_feature_vector(
		int32_t feat_idx, 
		CStructuredData* y)
{
	SG_ERROR("compute_joint_feature(int32_t, CStructuredData*) is not "
		 "implemented for %s!\n", get_name());

	return SGVector< float64_t >();
}

float64_t CStructuredModel::delta_loss(int32_t ytrue_idx, CStructuredData* ypred)
{
	if ( ytrue_idx < 0 || ytrue_idx >= m_labels->get_num_labels() )
		SG_ERROR("The label index must be inside [0, num_labels-1]\n");

	return delta_loss(m_labels->get_label(ytrue_idx), ypred);
}

float64_t CStructuredModel::delta_loss(CStructuredData* y1, CStructuredData* y2)
{
	SG_ERROR("delta_loss(CStructuredData*, CStructuredData*) is not "
		 "implemented for %s!\n", get_name());

	return 0.0;
}

void CStructuredModel::init()
{
	SG_ADD((CSGObject**) &m_labels, "m_labels", "Structured labels", 
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &m_features, "m_features", "Feature vectors", 
			MS_NOT_AVAILABLE);

	m_features = NULL;
	m_labels   = NULL;
}

bool CStructuredModel::check_training_setup() const
{
	// Nothing to do here
	return true;
}
