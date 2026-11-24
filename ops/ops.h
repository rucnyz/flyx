// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Float64 reference for every op the engine uses, one directory per op.  These
// define what each op means; the cpu_ref backend is built from them and every
// device kernel is tested against them.  Written for clarity, not speed.
#pragma once

#include "ops/dtype/ref.h"
#include "ops/linear/ref.h"
#include "ops/norm/ref.h"
#include "ops/activation/ref.h"
#include "ops/rope/ref.h"
#include "ops/attention/ref.h"
#include "ops/gdn/ref.h"
#include "ops/grouped_conv/ref.h"
#include "ops/sampling/ref.h"
#include "ops/draft/ref.h"
#include "ops/dspine/ref.h"
#include "ops/canvas/ref.h"
#include "ops/vision/ref.h"
