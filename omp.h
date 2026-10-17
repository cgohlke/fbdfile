/* Empty OpenMP definitions. */

typedef struct {} omp_lock_t;

void omp_destroy_lock(omp_lock_t *) {}

void omp_init_lock(omp_lock_t *) {}

void omp_set_lock(omp_lock_t *) {}

void omp_unset_lock(omp_lock_t *) {}
