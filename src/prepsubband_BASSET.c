#include <limits.h>
#include <ctype.h>
#include "presto.h"
#include "prepsubband_BASSET_cmd.h"
#include "mask.h"
#include "backend_common.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


// Use OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

#define RAWDATA (cmd->filterbankP || cmd->psrfitsP)

/* This causes the barycentric motion to be calculated once per TDT sec */
#define TDT 20.0

/* Simple linear interpolation macro */
#define LININTERP(X, xlo, xhi, ylo, yhi) ((ylo)+((X)-(xlo))*((yhi)-(ylo))/((xhi)-(xlo)))

/* Round a double or float to the nearest integer. */
/* x.5s get rounded away from zero.                */
#define NEAREST_LONG(x) (long) (x < 0 ? ceil(x - 0.5) : floor(x + 0.5))

static void write_data(FILE * outfiles[], int numfiles, float **outdata,
                       int startpoint, int numtowrite);
static void write_subs(FILE * outfiles[], int numfiles, short **subsdata,
                       int startpoint, int numtowrite);
static void write_padding(FILE * outfiles[], int numfiles, float value,
                          int numtowrite);
static int read_PRESTO_subbands(FILE * infiles[], int numfiles,
                                float *subbanddata, double timeperblk,
                                int *maskchans, int *nummasked, mask * obsmask,
                                float clip_sigma, float *padvals);
static int get_data(float **outdata, int blocksperread,
                    struct spectra_info *s,
                    mask * obsmask, int *idispdts, int **offsets,
                    int *padding, short **subsdata, float **outdata_BASSET, int totwrote, FILE **filter_inf_files);
static void update_infodata(infodata * idata, long datawrote, long padwrote,
                            int *barybins, int numbarybins, int downsamp);
static void print_percent_complete(int current, int number);
/* update code BASSET */
void float_dedisp_waterfall(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result);
void normalize(float *result, int numpts, int numchan);
int get_one_ms_Len(double downsampled_dt, int dsworklen);
void dsamp_in_time(float* arr, int numchan, int dsamp_len, float* downsamp_arr);
int compare_floats(const void* a, const void* b);
float calculate_mean(const float* data, int size);
float calculate_median(const float* data, int size);
float calculate_standard_deviation(const float *arr, int size);
float calculate_max(const float *arr, int size);
float calculate_sum(const float* arr, int size);
void calculate_matrix_sum(float **data, int m, int n, int dimension, int low_index, int high_index, float *result);
void boxcar_correlation(const float* arr, int numchan, int* BASSET_Ls, int BASSET_Ls_num, float** convolution_result);
float get_SNR(const float *a, int a_length, const float *a_bg, int a_bg_length);
int compare_arrays(const float *a, const float noise, int size);
int trigger(float **corr, float **corr_bg, int numchan, const int *BASSET_Ls, int BASSET_Ls_num);
void lm_gaussian(double *p, const double *t, const double *y_dat, int n);
void solve_linear_system(double* JtJ, double* g, double lambda, double* h);
void lm_matx_gaussian(const double *t, const double *p, const double *y_dat, 
                      double *JtJ, double *JtDy, double *X2, double *y_hat, 
                      double *J, int n);
void compute_jacobian(const double *t, const double *p, double *J, int n);
void gaussian(const double *t, const double *p, double *result, int n);
int fit_gaussian(const float* y_input, int n);
void convolve(const float *input, size_t input_size, const float *kernel, size_t kernel_size, float *output);
void autocorrelation(const float *input, size_t input_size, float *output);
size_t argmax(const float *array, size_t size);
int find_right_range(float** matrix_normalized_work, int work_length, int numchan, int BASSET_minL, int min_BASSET_Ls);
int find_left_range(float** matrix_normalized_work, int work_length, int numchan, int BASSET_minL, int min_BASSET_Ls);
void find_frequency_range(float **data_pulse, int data_pulse_length, float **data_bg, 
                int data_bg_length, int numchan, int frequency_low, 
                int frequency_high, int *result_low, int *result_high);
void adaptive_filter(float *data, int numpts, int numchan, int *BASSET_Ls, int BASSET_Ls_num, int BASSET_minL, int one_ms_len,
 float* outdata_BASSET, int totwrote, int downsamp, float dt, FILE* filter_inf_file);
void string_to_array(const char *str, int *array, int *len);
float sum_of_squares_sqrt(float arr[], int a, int b);
void get_stds(float** data_bg, int rows, int cols, float* stds);
void multi_trigger(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, 
                   int *BASSET_Ls, int BASSET_Ls_num, _Bool *trigger_flag);
void multi_trigger_else(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, _Bool *trigger_flag, int BASSET_minL, int min_BASSET_Ls);
void multi_range_search(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, _Bool *trigger_flag, int BASSET_minL, int min_BASSET_Ls);

// debug
void save_array_to_txt(float** array, int m, int n, const char* filename);

/* From CLIG */
static int insubs = 0;
static Cmdline *cmd;

#ifdef USEDMALLOC
#include "dmalloc.h"
#endif

int main(int argc, char *argv[])
{
    /* Any variable that begins with 't' means topocentric */
    /* Any variable that begins with 'b' means barycentric */
    FILE **outfiles;
    float **outdata = NULL;
    short **subsdata = NULL;
    double dtmp, *dms = NULL, avgdm = 0.0, maxdm, dsdt = 0;
    double tlotoa = 0.0, blotoa = 0.0, BW_ddelay = 0.0;
    double max = -9.9E30, min = 9.9E30, var = 0.0, avg = 0.0;
    double *btoa = NULL, *ttoa = NULL, avgvoverc = 0.0;
    char obs[3], ephem[10], rastring[50], decstring[50];
    long totnumtowrite, totwrote = 0, padwrote = 0, datawrote = 0;
    int **offsets;
    int ii, jj, numadded = 0, numremoved = 0, padding = 0;
    int numbarypts = 0, blocksperread = 0, worklen = 0;
    int numread = 0, numtowrite = 0;
    int padtowrite = 0, statnum = 0, good_padvals = 0;
    int numdiffbins = 0, *diffbins = NULL, *diffbinptr = NULL;
    int *idispdt;
    char *datafilenm;
    int dmprecision = 2;
    struct spectra_info s;
    infodata idata;
    mask obsmask;
    /* update code BASSET*/
    FILE **outfiles_BASSET = NULL;
    int BASSET_Ls_list[50];
    int BASSET_Ls_num[1];
    float **outdata_BASSET = NULL;
    FILE **filter_inf_files = NULL;

    /* Call usage() if we have no command line arguments */

    if (argc == 1) {
        Program = argv[0];
        printf("\n");
        usage();
        exit(0);
    }

    /* Parse the command line using the excellent program Clig */

    cmd = parseCmdline(argc, argv);
    spectra_info_set_defaults(&s);
    dmprecision = cmd->dmprec;
    s.filenames = cmd->argv;
    s.num_files = cmd->argc;
    // If we are zeroDMing, make sure that clipping is off.
    if (cmd->zerodmP)
        cmd->noclipP = 1;
    s.clip_sigma = cmd->clip;
    // -1 causes the data to determine if we use weights, scales, &
    // offsets for PSRFITS or flip the band for any data type where
    // we can figure that out with the data
    s.apply_flipband = (cmd->invertP) ? 1 : -1;
    s.apply_weight = (cmd->noweightsP) ? 0 : -1;
    s.apply_scale = (cmd->noscalesP) ? 0 : -1;
    s.apply_offset = (cmd->nooffsetsP) ? 0 : -1;
    s.remove_zerodm = (cmd->zerodmP) ? 1 : 0;
    if (cmd->noclipP) {
        cmd->clip = 0.0;
        s.clip_sigma = 0.0;
    }
    if (cmd->ifsP) {
        // 0 = default or summed, 1-4 are possible also
        s.use_poln = cmd->ifs + 1;
    }
    if (!cmd->numoutP)
        cmd->numout = LONG_MAX;

    if (cmd->ncpus > 1) {
#ifdef _OPENMP
        int maxcpus = omp_get_num_procs();
        int openmp_numthreads = (cmd->ncpus <= maxcpus) ? cmd->ncpus : maxcpus;
        // Make sure we are not dynamically setting the number of threads
        omp_set_dynamic(0);
        omp_set_num_threads(openmp_numthreads);
        printf("Using %d threads with OpenMP\n\n", openmp_numthreads);
#endif
    } else {
#ifdef _OPENMP
        omp_set_num_threads(1); // Explicitly turn off OpenMP
#endif
    }

#ifdef DEBUG
    showOptionValues();
#endif

    printf("\n\n");
    printf("          Pulsar Subband De-dispersion Routine\n");
    printf("                 by Scott M. Ransom\n\n");

    if (RAWDATA) {
        if (cmd->filterbankP)
            s.datatype = SIGPROCFB;
        else if (cmd->psrfitsP)
            s.datatype = PSRFITS;
    } else {                    // Attempt to auto-identify the data
        identify_psrdatatype(&s, 1);
        if (s.datatype == SIGPROCFB)
            cmd->filterbankP = 1;
        else if (s.datatype == PSRFITS)
            cmd->psrfitsP = 1;
        else if (s.datatype == SUBBAND)
            insubs = 1;
        else {
            printf
                ("Error:  Unable to identify input data files.  Please specify type.\n\n");
            exit(1);
        }
    }

    if (!RAWDATA)
        s.files = (FILE **) malloc(sizeof(FILE *) * s.num_files);
    if (RAWDATA || insubs) {
        char description[40];
        psrdatatype_description(description, s.datatype);
        if (s.num_files > 1)
            printf("Reading %s data from %d files:\n", description, s.num_files);
        else
            printf("Reading %s data from 1 file:\n", description);
        for (ii = 0; ii < s.num_files; ii++) {
            printf("  '%s'\n", cmd->argv[ii]);
            if (insubs)
                s.files[ii] = chkfopen(s.filenames[ii], "rb");
        }
        printf("\n");
        if (RAWDATA) {
            read_rawdata_files(&s);
            // Make sure that the requested number of subbands divides into the
            // the raw number of channels.
            if (s.num_channels % cmd->nsub) {
                printf("Error:  The number of subbands (-nsub %d) must divide into the\n"
                       "        number of channels (%d)\n\n",
                       cmd->nsub, s.num_channels);
                exit(1);
            }
            if (cmd->ignorechanstrP) {
                s.ignorechans = get_ignorechans(cmd->ignorechanstr, 0, s.num_channels-1,
                                                &s.num_ignorechans, &s.ignorechans_str);
                if (s.ignorechans_str==NULL) {
                    s.ignorechans_str = (char *)malloc(strlen(cmd->ignorechanstr)+1);
                    strcpy(s.ignorechans_str, cmd->ignorechanstr);
                }
            }
            print_spectra_info_summary(&s);
            spectra_info_to_inf(&s, &idata);
        } else {                // insubs
            cmd->nsub = s.num_files;
            s.N = chkfilelen(s.files[0], sizeof(short));
            s.padvals = gen_fvect(s.num_files);
            for (ii = 0; ii < s.num_files; ii++)
                s.padvals[ii] = 0.0;
            s.start_MJD = (long double *) malloc(sizeof(long double));
            s.start_spec = (long long *) malloc(sizeof(long long));
            s.num_spec = (long long *) malloc(sizeof(long long));
            s.num_pad = (long long *) malloc(sizeof(long long));
            s.start_spec[0] = 0L;
            s.num_spec[0] = s.N;
            s.num_pad[0] = 0L;
        }
        /* Read an input mask if wanted */
        if (cmd->maskfileP) {
            read_mask(cmd->maskfile, &obsmask);
            printf("Read mask information from '%s'\n\n", cmd->maskfile);
            good_padvals = determine_padvals(cmd->maskfile, &obsmask, s.padvals);
        } else {
            obsmask.numchan = obsmask.numint = 0;
        }
    }

    if (insubs) {
        char *root, *suffix;
        if (split_root_suffix(s.filenames[0], &root, &suffix) == 0) {
            printf("Error:  The input filename (%s) must have a suffix!\n\n",
                   s.filenames[0]);
            exit(1);
        }
        if (strncmp(suffix, "sub", 3) == 0) {
            char *tmpname;
            tmpname = calloc(strlen(root) + 10, 1);
            sprintf(tmpname, "%s.sub", root);
            readinf(&idata, tmpname);
            free(tmpname);
            s.num_channels = idata.num_chan;
            s.start_MJD[0] = idata.mjd_i + idata.mjd_f;
            s.dt = idata.dt;
            s.T = s.N * s.dt;
            s.lo_freq = idata.freq;
            s.df = idata.chan_wid;
            s.hi_freq = s.lo_freq + (s.num_channels - 1.0) * s.df;
            s.BW = s.num_channels * s.df;
            s.fctr = s.lo_freq - 0.5 * s.df + 0.5 * s.BW;
            s.spectra_per_subint = SUBSBLOCKLEN;
            print_spectra_info_summary(&s);
        } else {
            printf("\nThe input files (%s) must be subbands!  (i.e. *.sub##)\n\n",
                   cmd->argv[0]);
            exit(1);
        }
        free(root);
        free(suffix);
    }

    /* Determine the output file names and open them */

    datafilenm = (char *) calloc(strlen(cmd->outfile) + 20, 1);
    if (!cmd->subP) {
        printf("Writing output data to:\n");
        outfiles = (FILE **) malloc(cmd->numdms * sizeof(FILE *));
        dms = gen_dvect(cmd->numdms);
        for (ii = 0; ii < cmd->numdms; ii++) {
            dms[ii] = cmd->lodm + ii * cmd->dmstep;
            avgdm += dms[ii];
            sprintf(datafilenm, "%s_DM%.*f.dat", cmd->outfile, dmprecision, dms[ii]);
            outfiles[ii] = chkfopen(datafilenm, "wb");
            printf("   '%s'\n", datafilenm);
        }
        avgdm /= cmd->numdms;
        maxdm = dms[cmd->numdms - 1];
        /* update code BASSET*/
        if (cmd->BASSET_LsP){
            string_to_array(cmd-> BASSET_Ls, BASSET_Ls_list, BASSET_Ls_num);
            outfiles_BASSET = (FILE **) malloc(cmd->numdms * sizeof(FILE *));
            filter_inf_files = (FILE **) malloc(cmd->numdms * sizeof(FILE *));
            for (ii = 0; ii < cmd->numdms; ii++) {
                sprintf(datafilenm, "%s_DM%.*f_BASSET.filterinf", cmd->outfile, dmprecision, dms[ii]);
                filter_inf_files[ii] = chkfopen(datafilenm, "wb");
                sprintf(datafilenm, "%s_DM%.*f_BASSET.dat", cmd->outfile, dmprecision, dms[ii]);
                outfiles_BASSET[ii] = chkfopen(datafilenm, "wb");
		        printf("   '%s'\n", datafilenm);
            }
        }

    } else {
        char format_str[30];
        int num_places;

        if (!cmd->nobaryP) {
            printf("\nWarning:  You cannot (currently) barycenter subbands.\n"
                   "          Setting the '-nobary' flag automatically.\n");
            cmd->nobaryP = 1;
        }
        printf("Writing subbands to:\n");
        cmd->numdms = 1;
        dms = gen_dvect(cmd->numdms);
        dms[0] = cmd->subdm;
        cmd->lodm = cmd->subdm;
        avgdm = cmd->subdm;
        maxdm = cmd->subdm;
        outfiles = (FILE **) malloc(cmd->nsub * sizeof(FILE *));
        num_places = (int) ceil(log10(cmd->nsub));
        sprintf(format_str, "%%s_DM%%.*f.sub%%0%dd", num_places);
        for (ii = 0; ii < cmd->nsub; ii++) {
            sprintf(datafilenm, format_str, cmd->outfile, dmprecision, avgdm, ii);
            outfiles[ii] = chkfopen(datafilenm, "wb");
            printf("   '%s'\n", datafilenm);
        }
    }

    /* Set a few other key values */
    if (insubs)
        avgdm = idata.dm;
    if (RAWDATA)
        idata.dm = avgdm;
    dsdt = cmd->downsamp * idata.dt;
    BW_ddelay = delay_from_dm(maxdm, idata.freq) -
        delay_from_dm(maxdm, idata.freq + (idata.num_chan - 1) * idata.chan_wid);
    blocksperread = ((int) (BW_ddelay / idata.dt) / s.spectra_per_subint + 1);
    worklen = s.spectra_per_subint * blocksperread;
    /* The number of topo to bary time points to generate with TEMPO */
    numbarypts = (int) (s.T * 1.1 / TDT + 5.5) + 1;

    // Identify the TEMPO observatory code
    {
        char *outscope = (char *) calloc(40, sizeof(char));
        telescope_to_tempocode(idata.telescope, outscope, obs);
        free(outscope);
    }

    /* If we are offsetting into the file, change inf file start time */
    if (cmd->start > 0.0 || cmd->offset > 0) {
        if (cmd->start > 0.0) /* Offset in units of worklen */
            cmd->offset = (long) (cmd->start *
                                  idata.N / worklen) * worklen;
        add_to_inf_epoch(&idata, cmd->offset * idata.dt);
        printf("Offsetting into the input files by %ld spectra (%.6g sec)\n",
               cmd->offset, cmd->offset * idata.dt);
        if (RAWDATA)
            offset_to_spectra(cmd->offset, &s);
        else { // subbands
            for (ii = 0; ii < s.num_files; ii++)
                chkfileseek(s.files[ii], cmd->offset, sizeof(short), SEEK_SET);
            if (cmd->maskfileP)
                printf("WARNING!:  masking does not work with old-style subbands and -start or -offset!\n");
        }
    }

    if (cmd->nsub > s.num_channels) {
        printf
            ("Warning:  The number of requested subbands (%d) is larger than the number of channels (%d).\n",
             cmd->nsub, s.num_channels);
        printf("          Re-setting the number of subbands to %d.\n\n",
               s.num_channels);
        cmd->nsub = s.num_channels;
    }

    if (s.spectra_per_subint % cmd->downsamp) {
        printf
            ("Error:  The downsample factor (%d) must be a factor of the\n",
             cmd->downsamp);
        printf("        blocklength (%d).  Exiting.\n\n", s.spectra_per_subint);
        exit(1);
    }

    tlotoa = idata.mjd_i + idata.mjd_f; /* Topocentric epoch */

    /* Set the output length to a good number if it wasn't requested */
    if (!cmd->numoutP && !cmd->subP) {
        cmd->numoutP = 1;
        cmd->numout = choose_good_N((long long)(idata.N/cmd->downsamp));
        printf("Setting a 'good' output length of %ld samples\n", cmd->numout);
    }
    if (cmd->subP && (cmd->numout > idata.N/cmd->downsamp))
        cmd->numout = (long long)(idata.N/cmd->downsamp); // Don't pad subbands
    totnumtowrite = cmd->numout;

    if (cmd->nobaryP) {         /* Main loop if we are not barycentering... */
        double *dispdt;

        /* Dispersion delays (in bins).  The high freq gets no delay   */
        /* All other delays are positive fractions of bin length (dt)  */

        dispdt = subband_search_delays(s.num_channels, cmd->nsub, avgdm,
                                       idata.freq, idata.chan_wid, 0.0);
        idispdt = gen_ivect(s.num_channels);
        for (ii = 0; ii < s.num_channels; ii++)
            idispdt[ii] = NEAREST_LONG(dispdt[ii] / idata.dt);
        vect_free(dispdt);

        /* The subband dispersion delays (see note above) */

        offsets = gen_imatrix(cmd->numdms, cmd->nsub);
        for (ii = 0; ii < cmd->numdms; ii++) {
            double *subdispdt;

            subdispdt = subband_delays(s.num_channels, cmd->nsub, dms[ii],
                                       idata.freq, idata.chan_wid, 0.0);
            dtmp = subdispdt[cmd->nsub - 1];
            for (jj = 0; jj < cmd->nsub; jj++)
                offsets[ii][jj] = NEAREST_LONG((subdispdt[jj] - dtmp) / dsdt);
            vect_free(subdispdt);
        }

        /* Allocate our data array and start getting data */

        printf("\nDe-dispersing using:\n");
        printf("       Subbands = %d\n", cmd->nsub);
        printf("     Average DM = %.7g\n", avgdm);
        if (cmd->downsamp > 1) {
            printf("     Downsample = %d\n", cmd->downsamp);
            printf("  New sample dt = %.10g\n", dsdt);
        }
        printf("\n");

        if (cmd->subP)
            subsdata = gen_smatrix(cmd->nsub, worklen / cmd->downsamp);
        else{
            outdata = gen_fmatrix(cmd->numdms, worklen / cmd->downsamp);
            /* update code BASSET*/
            if (cmd->BASSET_LsP){
                outdata_BASSET = gen_fmatrix(cmd->numdms, worklen / cmd->downsamp);
            }
	}
        /* update code BASSET*/
        numread = get_data(outdata, blocksperread, &s,
                           &obsmask, idispdt, offsets, &padding, subsdata, outdata_BASSET, totwrote, filter_inf_files);
        while (numread == worklen) {

            numread /= cmd->downsamp;
            print_percent_complete(totwrote, totnumtowrite);

            /* Write the latest chunk of data, but don't   */
            /* write more than cmd->numout points.         */

            numtowrite = numread;
            if ((totwrote + numtowrite) > cmd->numout)
                numtowrite = cmd->numout - totwrote;
            if (cmd->subP)
                write_subs(outfiles, cmd->nsub, subsdata, 0, numtowrite);
            else{
                write_data(outfiles, cmd->numdms, outdata, 0, numtowrite);
                /* update code BASSET*/
                if(cmd->BASSET_LsP){
                    write_data(outfiles_BASSET, cmd->numdms, outdata_BASSET, 0, numtowrite);
                }
	    }
            totwrote += numtowrite;

            /* Update the statistics */

            if (!padding && !cmd->subP) {
                for (ii = 0; ii < numtowrite; ii++)
                    update_stats(statnum + ii, outdata[0][ii], &min, &max, &avg,
                                 &var);
                statnum += numtowrite;
            }

            /* Stop if we have written out all the data we need to */

            if (totwrote == cmd->numout)
                break;

            numread = get_data(outdata, blocksperread, &s,
                               &obsmask, idispdt, offsets, &padding, subsdata, outdata_BASSET, totwrote, filter_inf_files);
        }
        datawrote = totwrote;

    } else {                    /* Main loop if we are barycentering... */
        double maxvoverc = -1.0, minvoverc = 1.0, *voverc = NULL;
        double *dispdt;

        /* What ephemeris will we use?  (Default is DE405) */
        strcpy(ephem, "DE405");

        /* Define the RA and DEC of the observation */

        ra_dec_to_string(rastring, idata.ra_h, idata.ra_m, idata.ra_s);
        ra_dec_to_string(decstring, idata.dec_d, idata.dec_m, idata.dec_s);

        /* Allocate some arrays */

        btoa = gen_dvect(numbarypts);
        ttoa = gen_dvect(numbarypts);
        voverc = gen_dvect(numbarypts);
        for (ii = 0; ii < numbarypts; ii++)
            ttoa[ii] = tlotoa + TDT * ii / SECPERDAY;

        /* Call TEMPO for the barycentering */

        printf("\nGenerating barycentric corrections...\n");
        barycenter(ttoa, btoa, voverc, numbarypts, rastring, decstring, obs, ephem);
        for (ii = 0; ii < numbarypts; ii++) {
            if (voverc[ii] > maxvoverc)
                maxvoverc = voverc[ii];
            if (voverc[ii] < minvoverc)
                minvoverc = voverc[ii];
            avgvoverc += voverc[ii];
        }
        avgvoverc /= numbarypts;
        vect_free(voverc);
        blotoa = btoa[0];

        printf("   Average topocentric velocity (c) = %.7g\n", avgvoverc);
        printf("   Maximum topocentric velocity (c) = %.7g\n", maxvoverc);
        printf("   Minimum topocentric velocity (c) = %.7g\n\n", minvoverc);
        printf("De-dispersing and barycentering using:\n");
        printf("       Subbands = %d\n", cmd->nsub);
        printf("     Average DM = %.7g\n", avgdm);
        if (cmd->downsamp > 1) {
            printf("     Downsample = %d\n", cmd->downsamp);
            printf("  New sample dt = %.10g\n", dsdt);
        }
        printf("\n");

        /* Dispersion delays (in bins).  The high freq gets no delay   */
        /* All other delays are positive fractions of bin length (dt)  */

        dispdt = subband_search_delays(s.num_channels, cmd->nsub, avgdm,
                                       idata.freq, idata.chan_wid, avgvoverc);
        idispdt = gen_ivect(s.num_channels);
        for (ii = 0; ii < s.num_channels; ii++)
            idispdt[ii] = NEAREST_LONG(dispdt[ii] / idata.dt);
        vect_free(dispdt);

        /* The subband dispersion delays (see note above) */

        offsets = gen_imatrix(cmd->numdms, cmd->nsub);
        for (ii = 0; ii < cmd->numdms; ii++) {
            double *subdispdt;

            subdispdt = subband_delays(s.num_channels, cmd->nsub, dms[ii],
                                       idata.freq, idata.chan_wid, avgvoverc);
            dtmp = subdispdt[cmd->nsub - 1];
            for (jj = 0; jj < cmd->nsub; jj++)
                offsets[ii][jj] = NEAREST_LONG((subdispdt[jj] - dtmp) / dsdt);
            vect_free(subdispdt);
        }

        /* Convert the bary TOAs to differences from the topo TOAs in */
        /* units of bin length (dt) rounded to the nearest integer.   */

        dtmp = (btoa[0] - ttoa[0]);
        for (ii = 0; ii < numbarypts; ii++)
            btoa[ii] = ((btoa[ii] - ttoa[ii]) - dtmp) * SECPERDAY / dsdt;

        {                       /* Find the points where we need to add or remove bins */

            int oldbin = 0, currentbin;
            double lobin, hibin, calcpt;

            numdiffbins = labs(NEAREST_LONG(btoa[numbarypts - 1])) + 1;
            diffbins = gen_ivect(numdiffbins);
            diffbinptr = diffbins;
            for (ii = 1; ii < numbarypts; ii++) {
                currentbin = NEAREST_LONG(btoa[ii]);
                if (currentbin != oldbin) {
                    if (currentbin > 0) {
                        calcpt = oldbin + 0.5;
                        lobin = (ii - 1) * TDT / dsdt;
                        hibin = ii * TDT / dsdt;
                    } else {
                        calcpt = oldbin - 0.5;
                        lobin = -((ii - 1) * TDT / dsdt);
                        hibin = -(ii * TDT / dsdt);
                    }
                    while (fabs(calcpt) < fabs(btoa[ii])) {
                        /* Negative bin number means remove that bin */
                        /* Positive bin number means add a bin there */
                        *diffbinptr = NEAREST_LONG(LININTERP(calcpt, btoa[ii - 1],
                                                             btoa[ii], lobin,
                                                             hibin));
                        diffbinptr++;
                        calcpt = (currentbin > 0) ? calcpt + 1.0 : calcpt - 1.0;
                    }
                    oldbin = currentbin;
                }
            }
            *diffbinptr = cmd->numout;  /* Used as a marker */
        }
        diffbinptr = diffbins;

        /* Now perform the barycentering */

        if (cmd->subP)
            subsdata = gen_smatrix(cmd->nsub, worklen / cmd->downsamp);
        else
            outdata = gen_fmatrix(cmd->numdms, worklen / cmd->downsamp);
        numread = get_data(outdata, blocksperread, &s,
                           &obsmask, idispdt, offsets, &padding, subsdata, outdata_BASSET, totwrote, filter_inf_files);

        while (numread == worklen) {    /* Loop to read and write the data */
            int numwritten = 0;
            double block_avg, block_var;

            numread /= cmd->downsamp;
            /* Determine the approximate local average */
            avg_var(outdata[0], numread, &block_avg, &block_var);
            print_percent_complete(totwrote, totnumtowrite);

            /* Simply write the data if we don't have to add or */
            /* remove any bins from this batch.                 */
            /* OR write the amount of data up to cmd->numout or */
            /* the next bin that will be added or removed.      */

            numtowrite = abs(*diffbinptr) - datawrote;
            if ((totwrote + numtowrite) > cmd->numout)
                numtowrite = cmd->numout - totwrote;
            if (numtowrite > numread)
                numtowrite = numread;
            if (cmd->subP)
                write_subs(outfiles, cmd->nsub, subsdata, 0, numtowrite);
            else
                write_data(outfiles, cmd->numdms, outdata, 0, numtowrite);
            datawrote += numtowrite;
            totwrote += numtowrite;
            numwritten += numtowrite;

            /* Update the statistics */

            if (!padding && !cmd->subP) {
                for (ii = 0; ii < numtowrite; ii++)
                    update_stats(statnum + ii, outdata[0][ii], &min, &max, &avg,
                                 &var);
                statnum += numtowrite;
            }

            if ((datawrote == abs(*diffbinptr)) && (numwritten != numread) && (totwrote < cmd->numout)) {       /* Add/remove a bin */
                int skip, nextdiffbin;

                skip = numtowrite;

                do {            /* Write the rest of the data after adding/removing a bin  */

                    if (*diffbinptr > 0) {
                        /* Add a bin */
                        write_padding(outfiles, cmd->numdms, block_avg, 1);
                        numadded++;
                        totwrote++;
                    } else {
                        /* Remove a bin */
                        numremoved++;
                        datawrote++;
                        numwritten++;
                        skip++;
                    }
                    diffbinptr++;

                    /* Write the part after the diffbin */

                    numtowrite = numread - numwritten;
                    if ((totwrote + numtowrite) > cmd->numout)
                        numtowrite = cmd->numout - totwrote;
                    nextdiffbin = abs(*diffbinptr) - datawrote;
                    if (numtowrite > nextdiffbin)
                        numtowrite = nextdiffbin;
                    if (cmd->subP)
                        write_subs(outfiles, cmd->nsub, subsdata, skip, numtowrite);
                    else
                        write_data(outfiles, cmd->numdms, outdata, skip, numtowrite);
                    numwritten += numtowrite;
                    datawrote += numtowrite;
                    totwrote += numtowrite;

                    /* Update the statistics and counters */

                    if (!padding && !cmd->subP) {
                        for (ii = 0; ii < numtowrite; ii++)
                            update_stats(statnum + ii, outdata[0][skip + ii],
                                         &min, &max, &avg, &var);
                        statnum += numtowrite;
                    }
                    skip += numtowrite;

                    /* Stop if we have written out all the data we need to */

                    if (totwrote == cmd->numout)
                        break;
                } while (numwritten < numread);
            }
            /* Stop if we have written out all the data we need to */

            if (totwrote == cmd->numout)
                break;

            numread = get_data(outdata, blocksperread, &s,
                               &obsmask, idispdt, offsets, &padding, subsdata, outdata_BASSET, totwrote, filter_inf_files);
        }
    }

    /* Calculate the amount of padding we need (don't pad subbands) */

    if (!cmd->subP && (cmd->numout > totwrote))
        padwrote = padtowrite = cmd->numout - totwrote;

    /* Write the new info file for the output data */

    idata.dt = dsdt;
    update_infodata(&idata, totwrote, padtowrite, diffbins,
                    numdiffbins, cmd->downsamp);
    for (ii = 0; ii < cmd->numdms; ii++) {
        idata.dm = dms[ii];
        if (!cmd->nobaryP) {
            double baryepoch, barydispdt, baryhifreq;

            baryhifreq = idata.freq + (s.num_channels - 1) * idata.chan_wid;
            barydispdt = delay_from_dm(dms[ii], doppler(baryhifreq, avgvoverc));
            baryepoch = blotoa - (barydispdt / SECPERDAY);
            idata.bary = 1;
            idata.mjd_i = (int) floor(baryepoch);
            idata.mjd_f = baryepoch - idata.mjd_i;
        }
        if (cmd->subP)
            sprintf(idata.name, "%s_DM%.*f.sub", cmd->outfile, dmprecision, dms[ii]);
        else
            sprintf(idata.name, "%s_DM%.*f", cmd->outfile, dmprecision, dms[ii]);
        writeinf(&idata);

        /* update code BASSET*/
        if (cmd->BASSET_LsP)
        {
            sprintf(idata.name, "%s_DM%.*f_BASSET", cmd->outfile, dmprecision, dms[ii]);
            writeinf(&idata);
        }
        
        
    }

    /* Set the padded points equal to the average data point */

    if (idata.numonoff >= 1) {
        int index, startpad, endpad;

        for (ii = 0; ii < cmd->numdms; ii++) {
            fclose(outfiles[ii]);
            sprintf(datafilenm, "%s_DM%.*f.dat", cmd->outfile, dmprecision, dms[ii]);
            outfiles[ii] = chkfopen(datafilenm, "rb+");

            /* update code BASSET*/
            if (cmd->BASSET_LsP){
                fclose(outfiles_BASSET[ii]);
                sprintf(datafilenm, "%s_DM%.*f_BASSET.dat", cmd->outfile, dmprecision, dms[ii]);
                outfiles_BASSET[ii] = chkfopen(datafilenm, "rb+");
            }
            
        }
        for (ii = 0; ii < idata.numonoff; ii++) {
            index = 2 * ii;
            startpad = idata.onoff[index + 1];
            if (ii == idata.numonoff - 1)
                endpad = idata.N - 1;
            else
                endpad = idata.onoff[index + 2];
            for (jj = 0; jj < cmd->numdms; jj++){
                chkfseek(outfiles[jj], (startpad + 1) * sizeof(float), SEEK_SET);
		/* update code BASSET*/
                if (cmd->BASSET_LsP){
                    chkfseek(outfiles_BASSET[jj], (startpad + 1) * sizeof(float), SEEK_SET);
                }
	    }
            padtowrite = endpad - startpad;
            write_padding(outfiles, cmd->numdms, avg, padtowrite);

            /* update code BASSET*/
            if (cmd->BASSET_LsP){
                write_padding(outfiles_BASSET, cmd->numdms, avg, padtowrite);
            }
        }
    }

    /* Print simple stats and results */

    if (!cmd->subP) {
        var /= (datawrote - 1);
        print_percent_complete(1, 1);
        printf("\n\nDone.\n\nSimple statistics of the output data:\n");
        printf("             Data points written:  %ld\n", totwrote);
        if (padwrote)
            printf("          Padding points written:  %ld\n", padwrote);
        if (!cmd->nobaryP) {
            if (numadded)
                printf("    Bins added for barycentering:  %d\n", numadded);
            if (numremoved)
                printf("  Bins removed for barycentering:  %d\n", numremoved);
        }
        printf("           Maximum value of data:  %.2f\n", max);
        printf("           Minimum value of data:  %.2f\n", min);
        printf("              Data average value:  %.2f\n", avg);
        printf("         Data standard deviation:  %.2f\n", sqrt(var));
        printf("\n");
    } else {
        printf("\n\nDone.\n");
        printf("             Data points written:  %ld\n", totwrote);
        if (padwrote)
            printf("          Padding points written:  %ld\n", padwrote);
        if (!cmd->nobaryP) {
            if (numadded)
                printf("    Bins added for barycentering:  %d\n", numadded);
            if (numremoved)
                printf("  Bins removed for barycentering:  %d\n", numremoved);
        }
        printf("\n");
    }

    /* Close the files and cleanup */

    if (cmd->maskfileP) {
        free_mask(obsmask);
    }
    //  Close all the raw files and free their vectors
    close_rawfiles(&s);
    for (ii = 0; ii < cmd->numdms; ii++)
        fclose(outfiles[ii]);
    /* update code BASSET*/
    if (cmd->BASSET_LsP){
        vect_free(outdata_BASSET[0]);
        vect_free(outdata_BASSET);
        for (ii = 0; ii < cmd->numdms; ii++){
            fclose(outfiles_BASSET[ii]);
            fclose(filter_inf_files[ii]);
	}
    }
    if (cmd->subP) {
        vect_free(subsdata[0]);
        vect_free(subsdata);
    } else {
        vect_free(outdata[0]);
        vect_free(outdata);
    }
    free(outfiles);
    /* update code BASSET*/
    //free(outfiles_BASSET);
    //vect_free(dms);
    vect_free(idispdt);
    vect_free(offsets[0]);
    vect_free(offsets);
    if (!cmd->nobaryP) {
        vect_free(btoa);
        vect_free(ttoa);
        vect_free(diffbins);
    }
    
    return (0);
}

static void write_data(FILE * outfiles[], int numfiles, float **outdata,
                       int startpoint, int numtowrite)
{
    int ii;

    for (ii = 0; ii < numfiles; ii++)
        chkfwrite(outdata[ii] + startpoint, sizeof(float), numtowrite, outfiles[ii]);
}


static void write_subs(FILE * outfiles[], int numfiles, short **subsdata,
                       int startpoint, int numtowrite)
{
    int ii;

    for (ii = 0; ii < numfiles; ii++)
        chkfwrite(subsdata[ii] + startpoint, sizeof(short), numtowrite,
                  outfiles[ii]);
}


static void write_padding(FILE * outfiles[], int numfiles, float value,
                          int numtowrite)
{
    int ii;

    if (numtowrite <= 0) {
        return;
    } else if (numtowrite == 1) {
        for (ii = 0; ii < numfiles; ii++)
            chkfwrite(&value, sizeof(float), 1, outfiles[ii]);
    } else {
        int maxatonce = 8192, veclen, jj;
        float *buffer;
        veclen = (numtowrite > maxatonce) ? maxatonce : numtowrite;
        buffer = gen_fvect(veclen);
        for (ii = 0; ii < veclen; ii++)
            buffer[ii] = value;
        if (veclen == numtowrite) {
            for (ii = 0; ii < numfiles; ii++)
                chkfwrite(buffer, sizeof(float), veclen, outfiles[ii]);
        } else {
            for (ii = 0; ii < numtowrite / veclen; ii++) {
                for (jj = 0; jj < numfiles; jj++)
                    chkfwrite(buffer, sizeof(float), veclen, outfiles[jj]);
            }
            for (jj = 0; jj < numfiles; jj++)
                chkfwrite(buffer, sizeof(float), numtowrite % veclen, outfiles[jj]);
        }
        vect_free(buffer);
    }
}


static int read_PRESTO_subbands(FILE * infiles[], int numfiles,
                                float *subbanddata, double timeperblk,
                                int *maskchans, int *nummasked, mask * obsmask,
                                float clip_sigma, float *padvals)
/* Read short int subband data written by prepsubband */
{
    int ii, jj, index, numread = 0, mask = 0, offset;
    short subsdata[SUBSBLOCKLEN];
    double starttime, run_avg;
    float subband_sum;
    static int currentblock = 0;

    if (obsmask->numchan)
        mask = 1;

    /* Read the data */
    for (ii = 0; ii < numfiles; ii++) {
        numread = chkfread(subsdata, sizeof(short), SUBSBLOCKLEN, infiles[ii]);
        run_avg = 0.0;
        if (cmd->runavgP == 1) {
            for (jj = 0; jj < numread; jj++)
                run_avg += (float) subsdata[jj];
            run_avg /= numread;
        }
        for (jj = 0, index = ii; jj < numread; jj++, index += numfiles)
            subbanddata[index] = (float) subsdata[jj] - run_avg;
        for (jj = numread; jj < SUBSBLOCKLEN; jj++, index += numfiles)
            subbanddata[index] = 0.0;
    }

    if (mask) {
        starttime = currentblock * timeperblk;
        *nummasked = check_mask(starttime, timeperblk, obsmask, maskchans);
    }

    /* Clip nasty RFI if requested and we're not masking all the channels */
    if ((clip_sigma > 0.0) && !(mask && (*nummasked == -1))) {
        clip_times(subbanddata, SUBSBLOCKLEN, numfiles, clip_sigma, padvals);
    }

    /* Mask it if required */
    if (mask && numread) {
        if (*nummasked == -1) { /* If all channels are masked */
            for (ii = 0; ii < SUBSBLOCKLEN; ii++)
                memcpy(subbanddata + ii * numfiles, padvals,
                       sizeof(float) * numfiles);
        } else if (*nummasked > 0) {    /* Only some of the channels are masked */
            int channum;
            for (ii = 0; ii < SUBSBLOCKLEN; ii++) {
                offset = ii * numfiles;
                for (jj = 0; jj < *nummasked; jj++) {
                    channum = maskchans[jj];
                    subbanddata[offset + channum] = padvals[channum];
                }
            }
        }
    }

    /* Zero-DM removal if required */
    if (cmd->zerodmP == 1) {
        for (ii = 0; ii < SUBSBLOCKLEN; ii++) {
            offset = ii * numfiles;
            subband_sum = 0.0;
            for (jj = offset; jj < offset + numfiles; jj++) {
                subband_sum += subbanddata[jj];
            }
            subband_sum /= (float) numfiles;
            /* Remove the channel average */
            for (jj = offset; jj < offset + numfiles; jj++) {
                subbanddata[jj] -= subband_sum;
            }
        }
    }

    currentblock += 1;
    return numread;
}



static int get_data(float **outdata, int blocksperread,
                    struct spectra_info *s,
                    mask * obsmask, int *idispdts, int **offsets,
                    int *padding, short **subsdata, float **outdata_BASSET, int totwrote, FILE **filter_inf_files)
{
    static int firsttime = 1, *maskchans = NULL, blocksize;
    static int worklen, dsworklen, one_ms_len;
    static float *tempzz, *data1, *data2, *dsdata1 = NULL, *dsdata2 = NULL;
    static float *currentdata, *lastdata, *currentdsdata, *lastdsdata;
    static double blockdt;
    int totnumread = 0, numread = 0, ii, jj, tmppad = 0, nummasked = 0;
    /* update code BASSET */
    int BASSET_Ls_list[50];
    static int BASSET_Ls_num[1];
    static float *data_BASSET = NULL;
    
    if (firsttime) {
        if (cmd->maskfileP)
            maskchans = gen_ivect(s->num_channels);
        worklen = s->spectra_per_subint * blocksperread;
        dsworklen = worklen / cmd->downsamp;
        // Make sure that our working blocks are long enough...
        for (ii = 0; ii < cmd->numdms; ii++) {
            for (jj = 0; jj < cmd->nsub; jj++) {
                if (offsets[ii][jj] > dsworklen)
                    printf
                        ("WARNING!:  (offsets[%d][%d] = %d) > (dsworklen = %d)\n",
                         ii, jj, offsets[ii][jj], dsworklen);
            }
        }

        blocksize = s->spectra_per_subint * cmd->nsub;
        blockdt = s->spectra_per_subint * s->dt;
        data1 = gen_fvect(cmd->nsub * worklen);
        data2 = gen_fvect(cmd->nsub * worklen);
        currentdata = data1;
        lastdata = data2;
        if (cmd->downsamp > 1) {
            dsdata1 = gen_fvect(cmd->nsub * dsworklen);
            dsdata2 = gen_fvect(cmd->nsub * dsworklen);
            currentdsdata = dsdata1;
            lastdsdata = dsdata2;
        } else {
            currentdsdata = data1;
            lastdsdata = data2;
        }
    }
    while (1) {
        if (RAWDATA || insubs) {
            for (ii = 0; ii < blocksperread; ii++) {
                if (RAWDATA)
                    numread = read_subbands(currentdata + ii * blocksize, idispdts,
                                            cmd->nsub, s, 0, &tmppad,
                                            maskchans, &nummasked, obsmask);
                else if (insubs)
                    numread = read_PRESTO_subbands(s->files, s->num_files,
                                                   currentdata + ii * blocksize,
                                                   blockdt, maskchans, &nummasked,
                                                   obsmask, cmd->clip, s->padvals);
                if (!firsttime)
                    totnumread += numread;
                if (numread != s->spectra_per_subint) {
                    for (jj = ii * blocksize; jj < (ii + 1) * blocksize; jj++)
                        currentdata[jj] = 0.0;
                }
                if (tmppad)
                    *padding = 1;
            }
        }
        /* Downsample the subband data if needed */
        if (cmd->downsamp > 1) {
            int kk, index;
            float ftmp;
            for (ii = 0; ii < dsworklen; ii++) {
                const int dsoffset = ii * cmd->nsub;
                const int offset = dsoffset * cmd->downsamp;
                for (jj = 0; jj < cmd->nsub; jj++) {
                    const int dsindex = dsoffset + jj;
                    index = offset + jj;
                    currentdsdata[dsindex] = ftmp = 0.0;
                    for (kk = 0; kk < cmd->downsamp; kk++) {
                        ftmp += currentdata[index];
                        index += cmd->nsub;
                    }
                    currentdsdata[dsindex] += ftmp / cmd->downsamp;
                }
            }
        }
        if (firsttime) {
            SWAP(currentdata, lastdata);
            SWAP(currentdsdata, lastdsdata);
            firsttime = 0;
        } else
            break;
    }
    if (!cmd->subP) {
        for (ii = 0; ii < cmd->numdms; ii++){
            float_dedisp(currentdsdata, lastdsdata, dsworklen,
                         cmd->nsub, offsets[ii], 0.0, outdata[ii]);
            if(cmd->BASSET_LsP){
                one_ms_len = get_one_ms_Len(cmd->downsamp * s->dt, dsworklen);
                string_to_array(cmd-> BASSET_Ls, BASSET_Ls_list, BASSET_Ls_num);
                data_BASSET =  gen_fvect(cmd->nsub * dsworklen);
                float_dedisp_waterfall(currentdsdata, lastdsdata, dsworklen,
                                     cmd->nsub, offsets[ii], 0.0, data_BASSET);
                adaptive_filter(data_BASSET, dsworklen, cmd->nsub, BASSET_Ls_list, *BASSET_Ls_num, cmd->BASSET_minL,
                 one_ms_len, outdata_BASSET[ii], totwrote, cmd->downsamp, s->dt, filter_inf_files[ii]);
            }
	}
    } else {
        /* Input format is sub1[0], sub2[0], sub3[0], ..., sub1[1], sub2[1], sub3[1], ... */
        float infloat;
        for (ii = 0; ii < cmd->nsub; ii++) {
            for (jj = 0; jj < dsworklen; jj++) {
                infloat = lastdsdata[ii + (cmd->nsub * jj)];
                subsdata[ii][jj] = (short) (infloat + 0.5);
                //if ((float) subsdata[ii][jj] != infloat)
                //   printf
                //       ("Warning:  We are incorrectly converting subband data! float = %f  short = %d\n",
                //         infloat, subsdata[ii][jj]);
            }
        }
    }
    SWAP(currentdata, lastdata);
    SWAP(currentdsdata, lastdsdata);
    if (totnumread != worklen) {
        if (cmd->maskfileP)
            vect_free(maskchans);
        vect_free(data1);
        vect_free(data2);
        if (cmd->BASSET_LsP){
            vect_free(data_BASSET);
        }
        if (cmd->downsamp > 1) {
            vect_free(dsdata1);
            vect_free(dsdata2);
        }
    }
    return totnumread;
}


static void print_percent_complete(int current, int number)
{
    static int newper = 0, oldper = -1;

    newper = (int) (current / (float) (number) * 100.0);
    if (newper < 0)
        newper = 0;
    if (newper > 100)
        newper = 100;
    if (newper > oldper) {
        printf("\rAmount complete = %3d%%", newper);
        fflush(stdout);
        oldper = newper;
    }
}


static void update_infodata(infodata * idata, long datawrote, long padwrote,
                            int *barybins, int numbarybins, int downsamp)
/* Update our infodata for barycentering and padding */
{
    int ii, jj, index;

    idata->N = datawrote + padwrote;
    if (idata->numonoff == 0) {
        if (padwrote) {
            idata->numonoff = 2;
            idata->onoff[0] = 0.0;
            idata->onoff[1] = datawrote - 1;
            idata->onoff[2] = idata->N - 1;
            idata->onoff[3] = idata->N - 1;
        }
        return;
    } else {
        for (ii = 0; ii < idata->numonoff; ii++) {
            idata->onoff[ii * 2] /= downsamp;
            idata->onoff[ii * 2 + 1] /= downsamp;
        }
    }

    /* Determine the barycentric onoff bins (approximate) */

    if (numbarybins) {
        int numadded = 0, numremoved = 0;

        ii = 1;                 /* onoff index    */
        jj = 0;                 /* barybins index */
        while (ii < idata->numonoff * 2) {
            while (abs(barybins[jj]) <= idata->onoff[ii] && jj < numbarybins) {
                if (barybins[jj] < 0)
                    numremoved++;
                else
                    numadded++;
                jj++;
            }
            idata->onoff[ii] += numadded - numremoved;
            ii++;
        }
    }

    /* Now cut off the extra onoff bins */

    for (ii = 1, index = 1; ii <= idata->numonoff; ii++, index += 2) {
        if (idata->onoff[index - 1] > idata->N - 1) {
            idata->onoff[index - 1] = idata->N - 1;
            idata->onoff[index] = idata->N - 1;
            break;
        }
        if (idata->onoff[index] > datawrote - 1) {
            idata->onoff[index] = datawrote - 1;
            idata->numonoff = ii;
            if (padwrote) {
                idata->numonoff++;
                idata->onoff[index + 1] = idata->N - 1;
                idata->onoff[index + 2] = idata->N - 1;
            }
            break;
        }
    }
}


/* update code BASSET */
void float_dedisp_waterfall(float *data, float *lastdata,
                  int numpts, int numchan,
                  int *delays, float approx_mean, float *result)
{
    long long ii, jj, kk;

    for (ii = 0; ii < numpts; ii++){
        for (jj = 0; jj < numchan; jj++)
            result[numchan * ii + jj] = -approx_mean;
    }

    /* De-disperse */
    for (ii = 0; ii < numchan; ii++) {
        jj = ii + (long long)(delays[ii]) * numchan;
        for (kk = 0; kk < numpts - delays[ii]; kk++, jj += numchan){
            result[numchan * kk + ii] = lastdata[jj];
	}
        jj = ii;
        for (; kk < numpts; kk++, jj += numchan)
            result[numchan * kk + ii] = data[jj];
    }

    /* Normalize the channel */
    normalize(result, numpts, numchan);

}


void normalize(float *result, int numpts, int numchan) {
    float channel_means[numchan];

    for (int i = 0; i < numchan; i++) {
        float sum = 0;
        for (int j = 0; j < numpts; j++) {
            sum += result[j * numchan + i];
        }
        channel_means[i] = sum / numpts;
    }
    
    int non_zero_columns[numchan];
    int new_num_columns = 0;

    for (int i = 0; i < numchan; i++) {
        if (channel_means[i] != 0) {
            non_zero_columns[new_num_columns] = i;
            new_num_columns++;
        }
    }

    float new_channel_means[new_num_columns];
    for (int i = 0; i < new_num_columns; i++) {
        new_channel_means[i] = channel_means[non_zero_columns[i]];
    }

    float channel_means_std = calculate_standard_deviation(new_channel_means, new_num_columns);
    float channel_means_mean = calculate_mean(new_channel_means, new_num_columns);
    for (int i = 0; i < numchan; i++) {
        float deviation = channel_means[i] - channel_means_mean;
        if (fabs(deviation) < 3 * channel_means_std){
            for (int j = 0; j < numpts; j++) {
                result[j * numchan + i] -= channel_means[i];
            }
        }else{
            for (int j = 0; j < numpts; j++) {
                result[j * numchan + i] = 0;
            }
        }
    }
}


int get_one_ms_Len(double downsampled_dt, int dsworklen) {
    int one_ms_len = 1;

    while (1) {
        if (one_ms_len * downsampled_dt >= 0.0005 && dsworklen % one_ms_len == 0) {
            break;
        }
        one_ms_len++;
    }

    return one_ms_len;
}


void dsamp_in_time(float* arr, int numchan, int dsamp_len, float* downsamp_arr) {
    for (int i = 0; i < numchan; i++) {
        downsamp_arr[i] = 0;
        for (int j = 0; j < dsamp_len; j++) {
            downsamp_arr[i] += arr[i + j * numchan];
        }
    }
}


int compare_floats(const void* a, const void* b) {
    return (*(float*)a > *(float*)b) - (*(float*)a < *(float*)b);
}

float calculate_mean(const float* data, int size) {
    if (size == 0) return 0.0;
    float sum = 0.0;
    for (int i = 0; i < size; i++) sum += data[i];
    return sum / size;
}

float calculate_median(const float* data, int size) {
    if (size == 0) return 0.0;

    float* temp_data = (float*)malloc(size * sizeof(float));
    if (!temp_data) return 0.0;

    memcpy(temp_data, data, size * sizeof(float));

    int low = 0;
    int high = size - 1;
    int median_index = size / 2;
    int k1, k2;
    int found_k1 = 0, found_k2 = 0;

    if (size % 2 == 0) {
        // For even-sized arrays, we need the two middle elements
        k1 = median_index - 1;
        k2 = median_index;
    } else {
        // For odd-sized arrays, we need the single middle element
        k1 = k2 = median_index;
    }

    while (low <= high && (!found_k1 || !found_k2)) {
        float pivot = temp_data[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (temp_data[j] <= pivot) {
                i++;
                float temp = temp_data[i];
                temp_data[i] = temp_data[j];
                temp_data[j] = temp;
            }
        }

        float temp = temp_data[i + 1];
        temp_data[i + 1] = temp_data[high];
        temp_data[high] = temp;

        int pivotIndex = i + 1;

        if (pivotIndex == k1) {
            found_k1 = 1;
        }
        if (pivotIndex == k2) {
            found_k2 = 1;
        }

        if (found_k1 && found_k2) {
            break;
        } else if (pivotIndex < k1) {
            low = pivotIndex + 1;
        } else {
            high = pivotIndex - 1;
        }
    }

    float median;
    if (size % 2 == 0) {
        // For even-sized arrays, calculate the average of the two middle elements
        float lower = temp_data[k1];
        float upper = temp_data[k2];
        median = (lower + upper) / 2.0;
    } else {
        // For odd-sized arrays, return the middle element
        median = temp_data[k1];
    }

    free(temp_data);

    return median;
}


float calculate_standard_deviation(const float *arr, int size) {
    if (size <= 1) {
        return 0.0;
    }

    float mean = calculate_mean(arr, size);
    float sum_squared_diff = 0.0;

    for (int i = 0; i < size; ++i) {
        float diff = arr[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / (size - 1));
}


float calculate_max(const float *arr, int size) {
    if (size <= 0) {
        fprintf(stderr, "Invalid array size\n");
        return 0.0;
    }

    float max_value = arr[0];

    for (int i = 1; i < size; ++i) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    return max_value;
}

int calculate_min(int arr[], int n) {
    int min = arr[0]; 

    for (int i = 1; i < n; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    return min;
}

float calculate_sum(const float* arr, int size) {
    float result = 0;
    for (int i = 0; i < size; i++) {
        result += arr[i];
    }
    return result;
}


void calculate_matrix_sum(float **data, int m, int n, int dimension, int low_index, int high_index, float *result) {
    if (dimension < 0 || dimension > 1 || low_index < 0 || high_index > (dimension == 0 ? m : n)) {
	printf("%d %d %d %d \n ", low_index, high_index, m, n);
        fprintf(stderr, "Error: Invalid input parameters.\n");
        return;
    }

    int result_length = (dimension == 0) ? n : m;
    for (int i = 0; i < result_length; ++i) {
        result[i] = 0.0;
    }

    if (dimension == 0) {
        for (int i = low_index; i < high_index; ++i) {
            for (int j = 0; j < n; ++j) {
                result[j] += data[i][j];
            }
        }
    } else if (dimension == 1) {
        for (int i = 0; i < m; ++i) {
            for (int j = low_index; j < high_index; ++j) {
                result[i] += data[i][j];
            }
        }
    }
}


void boxcar_correlation(const float* arr, int numchan, int* BASSET_Ls, int BASSET_Ls_num, float** convolution_result) {
    float *sums = (float *)malloc((numchan + 1) * sizeof(float));
    sums[0] = 0.0;
    for (int ii = 0; ii < numchan; ++ii) {
        sums[ii + 1] = sums[ii] + arr[ii];
    }

    for (int jj = 0; jj < BASSET_Ls_num; ++jj) {
        int width = BASSET_Ls[jj];
        for (int ii = 0; ii < numchan - width + 1; ++ii) {
            convolution_result[jj][ii] = sums[ii + width] - sums[ii];
        }
    }
    free(sums);
}


float get_SNR(const float *a, int a_length, const float *a_bg, int a_bg_length) {

    float estimated_variance = calculate_standard_deviation(a_bg, a_bg_length);
    float estimated_mean = calculate_median(a_bg, a_bg_length);
    float max_value = calculate_max(a, a_length);
    float snr = (max_value - estimated_mean) / estimated_variance;
    return snr;
}


int compare_arrays(const float *a, const float noise, int size) {
    int count_within_tolerance = 0;

    for (int i = 0; i < size; ++i) {
        if (a[i] >= noise) {
            count_within_tolerance += 1;
        }
    }

    return count_within_tolerance;
}


int trigger(float **corr, float **corr_bg, int numchan, const int *BASSET_Ls, int BASSET_Ls_num) {
    float *SNRs = (float *)malloc(BASSET_Ls_num * sizeof(float));

    for (int jj = 0; jj < BASSET_Ls_num; ++jj) {
        int width = BASSET_Ls[jj];
        SNRs[jj] = get_SNR(corr[jj], numchan - width + 1, corr_bg[jj], numchan - width + 1);
    }
    int result = compare_arrays(&SNRs[1], SNRs[0], BASSET_Ls_num - 1);
    free(SNRs);

    return result >= 3 ? 1 : 0;
}


void gaussian(const double *t, const double *p, double *result, int n) {
    double a = p[0];
    double mu = p[1];
    double sigma = p[2];
    
    for (int i = 0; i < n; ++i) {
        result[i] = a * exp(-0.5 * pow((t[i] - mu) / sigma, 2));
    }
}


void compute_jacobian(const double *t, const double *p, double *J, int n) {
    double a = p[0];
    double mu = p[1];
    double sigma = p[2];
    
    for (int i = 0; i < n; ++i) {
        double exp_term = exp(-pow(t[i] - mu, 2) / (2 * sigma * sigma));
        J[i * 3] = exp_term;
        J[i * 3 + 1] = a * (t[i] - mu) / (sigma * sigma) * exp_term;
        J[i * 3 + 2] = a * (pow(t[i] - mu, 2) - sigma * sigma) / (sigma * sigma * sigma) * exp_term;
    }
}


void lm_matx_gaussian(const double *t, const double *p, const double *y_dat, 
                      double *JtJ, double *JtDy, double *X2, double *y_hat, 
                      double *J, int n) {
    double *residuals = (double *)malloc(n * sizeof(double));
    gaussian(t, p, y_hat, n);
    compute_jacobian(t, p, J, n);
    
    for (int i = 0; i < n; ++i) {
        residuals[i] = y_dat[i] - y_hat[i];
    }
    
    // Compute J^T * J
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            JtJ[i * 3 + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                JtJ[i * 3 + j] += J[k * 3 + i] * J[k * 3 + j];
            }
        }
    }
    
    // Compute J^T * residuals
    for (int i = 0; i < 3; ++i) {
        JtDy[i] = 0.0;
        for (int k = 0; k < n; ++k) {
            JtDy[i] += J[k * 3 + i] * residuals[k];
        }
    }
    
    // Compute Chi-squared error
    *X2 = 0.0;
    for (int i = 0; i < n; ++i) {
        *X2 += residuals[i] * residuals[i];
    }
    
    free(residuals);
}


void solve_linear_system(double* JtJ, double* g, double lambda, double* h) {
    // Size of the matrix
    int n = 3;
    
    // Create temporary arrays
    double* augmented_JtJ = (double*)malloc(n * n * sizeof(double));
    double* temp_g = (double*)malloc(n * sizeof(double));
    
    // Copy JtJ to augmented_JtJ and add lambda to the diagonal
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented_JtJ[i * n + j] = JtJ[i * n + j];
        }
        augmented_JtJ[i * n + i] += lambda * augmented_JtJ[i * n + i];
    }
    
    // Copy g to temp_g
    for (int i = 0; i < n; ++i) {
        temp_g[i] = g[i];
    }

    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double factor = augmented_JtJ[j * n + i] / augmented_JtJ[i * n + i];
            for (int k = i; k < n; ++k) {
                augmented_JtJ[j * n + k] -= factor * augmented_JtJ[i * n + k];
            }
            temp_g[j] -= factor * temp_g[i];
        }
    }

    // Back-substitution
    for (int i = n - 1; i >= 0; --i) {
        h[i] = temp_g[i];
        for (int j = i + 1; j < n; ++j) {
            h[i] -= augmented_JtJ[i * n + j] * h[j];
        }
        h[i] /= augmented_JtJ[i * n + i];
    }

    // Free allocated memory
    free(augmented_JtJ);
    free(temp_g);
}


void lm_gaussian(double *p, const double *t, const double *y_dat, int n) {
    double *JtJ = (double *)malloc(9 * sizeof(double));  // 3x3 matrix
    double *JtDy = (double *)malloc(3 * sizeof(double));  // Vector of length 3
    double *J = (double *)malloc(n * 3 * sizeof(double));  // Jacobian matrix (n x 3)
    double *y_hat = (double *)malloc(n * sizeof(double));   // Model predictions
    
    double X2, X2_old, lambda, rho;
    int iteration = 0;
    const double max_iter = 10000;
    const double epsilon_1 = 1e-3;
    const double epsilon_2 = 1e-3;
    const double epsilon_4 = 1e-1;
    const double lambda_0 = 1e-2;
    const double lambda_UP_FAC = 11;
    const double lambda_DN_FAC = 9;
    
    lambda = lambda_0;
    
    // Initial computation
    lm_matx_gaussian(t, p, y_dat, JtJ, JtDy, &X2, y_hat, J, n);
    X2_old = X2;
    
    while (iteration <= max_iter) {
        iteration++;
        
        // Solve (J^T * J + lambda * I) * h = J^T * dy
        double *h = (double *)malloc(3 * sizeof(double));      
        solve_linear_system(JtJ, JtDy, lambda, h);
        
        // Update parameters
        double *p_try = (double *)malloc(3 * sizeof(double));
        for (int i = 0; i < 3; ++i) {
            p_try[i] = p[i] + h[i];
        }
        
        // Compute new residuals
        double *delta_y = (double *)malloc(n * sizeof(double));
        gaussian(t, p_try, y_hat, n);
        for (int i = 0; i < n; ++i) {
            delta_y[i] = y_dat[i] - y_hat[i];
        }
        
        // Compute new Chi-squared
        double X2_try = 0.0;
        for (int i = 0; i < n; ++i) {
            X2_try += delta_y[i] * delta_y[i];
        }
        
        // Compute rho
        rho = (X2 - X2_try) / (0.5 * (lambda * (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]) + JtDy[0] * h[0] + JtDy[1] * h[1] + JtDy[2] * h[2]));
        if (rho > epsilon_4) {
            // Update parameters
            for (int i = 0; i < 3; ++i) {
                p[i] = p_try[i];
            }
            lm_matx_gaussian(t, p, y_dat, JtJ, JtDy, &X2, y_hat, J, n);
            // Update lambda
            lambda = fmax(lambda / lambda_DN_FAC, 1.e-7);
            X2_old = X2;
        } else {
            // Revert parameters
            X2 = X2_old;
            lambda = fmin(lambda * lambda_UP_FAC, 1.e7);
        }
        
        // Check convergence
        double max_abs_JtDy = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (fabs(JtDy[i]) > max_abs_JtDy) {
                max_abs_JtDy = fabs(JtDy[i]);
            }
        }
        
        if (max_abs_JtDy < epsilon_1 && iteration > 2) {
            break;
        }
        
        double max_rel_h = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (fabs(h[i]) / (fabs(p[i]) + 1e-12) > max_rel_h) {
                max_rel_h = fabs(h[i]) / (fabs(p[i]) + 1e-12);
            }
        }
        
        if (max_rel_h < epsilon_2 && iteration > 2) {
            break;
        }
        
        free(h);
        free(p_try);
        free(delta_y);
    }
    
    free(JtJ);
    free(JtDy);
    free(J);
    free(y_hat);
}


int fit_gaussian(const float* y_input, int n){
    double *t;
    double *y_dat = (double *)malloc(n * sizeof(double));

    t = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        t[i] = (double)i;
	y_dat[i] = (double)y_input[i];
    }
    double initial_p[3] = {calculate_max(y_input, n), argmax(y_input, n), 1.0}; // Initial guess for [a, mu, sigma]
    lm_gaussian(initial_p, t, y_dat, n);
    free(t);
    free(y_dat);
    int sigma = (int) initial_p[2];
    return sigma;
}


void convolve(const float *input, size_t input_size, const float *kernel, size_t kernel_size, float *output) {
    // Allocate memory for the convolution result
    float *temp_result = (float *)malloc(sizeof(float) * (input_size + kernel_size - 1));

    // Perform convolution
    for (size_t i = 0; i < input_size + kernel_size - 1; ++i) {
        temp_result[i] = 0.0;
        for (size_t j = 0; j < kernel_size; ++j) {
            if (i >= j && i - j < input_size) {
                temp_result[i] += input[i - j] * kernel[j];
            }
        }
    }

    // Copy the central part (same as 'same' mode in numpy.convolve)
    size_t start_index = (kernel_size - 1) / 2;
    for (size_t i = 0; i < input_size; ++i) {
        output[i] = temp_result[i + start_index];
    }

    // Free temporary memory
    free(temp_result);
}


void autocorrelation(const float *input, size_t input_size, float *output) {
    // Allocate memory for the result
    size_t result_size = 2 * input_size - 1;
    int bias;
    float *temp_result = (float *)malloc(sizeof(float) * result_size);
    
    // Perform correlation
    for (size_t i = 0; i < result_size; ++i) {
        temp_result[i] = 0.0;
        bias = i - input_size + 1;
        for (size_t j = 0; j < input_size - abs(bias); ++j){
            if(bias < 0){
                temp_result[i] += input[input_size - 1 - j] * input[input_size - 1 - j + bias];
            }else{
                temp_result[i] += input[input_size - 1 - j] * input[input_size - 1 - j - bias];
            }
        }
    }

    // Copy the result
    for (size_t i = 0; i < result_size; ++i) {
        output[i] = temp_result[i];
    }

    // Free temporary memory
    free(temp_result);
}


size_t argmax(const float *array, size_t size) {
    if (size == 0) {
        fprintf(stderr, "Error: Empty array.\n");
        return 0;
    }

    size_t max_index = 0;
    float max_value = array[0];

    for (size_t i = 1; i < size; ++i) {
        if (array[i] > max_value) {
            max_value = array[i];
            max_index = i;
        }
    }

    return max_index;
}


int find_right_range(float** matrix_normalized_work, int work_length, int numchan, int BASSET_minL, int min_BASSET_Ls){
    int right_range = 0, ii, jj, single_bin_width, real_width, single_bin_center, real_center, flag=0;
    float single_bin_array[numchan], single_bin_array_boxcar[numchan], singal_array[numchan], signal_array_boxcar[numchan];
    float boxcar_kernel[numchan];
    for (jj = 0; jj < numchan; jj++) {
        boxcar_kernel[jj] = 1.0;
    }
    float autocorrelation_result[2 * numchan - 1]; 
    while(right_range < work_length){
	    if (flag > 2){
            break;
        }
        for (ii = 0; ii < numchan; ii++) {
           single_bin_array[ii] = matrix_normalized_work[work_length + right_range][ii];
        }
	    convolve(single_bin_array, numchan, boxcar_kernel, BASSET_minL, single_bin_array_boxcar);
        autocorrelation(single_bin_array_boxcar, numchan, autocorrelation_result);
        single_bin_width = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
        if (single_bin_width > min_BASSET_Ls / 2){
            if (single_bin_width > numchan){
                single_bin_width = numchan;
            }
            convolve(single_bin_array, numchan, boxcar_kernel, single_bin_width, single_bin_array_boxcar);
            single_bin_center = argmax(single_bin_array_boxcar, numchan);
            calculate_matrix_sum(matrix_normalized_work, 2 * work_length, numchan, 0,
                             work_length, work_length + 1 + right_range, singal_array);
            convolve(singal_array, numchan, boxcar_kernel, BASSET_minL, signal_array_boxcar);
            autocorrelation(signal_array_boxcar, numchan, autocorrelation_result);
            real_width = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
            if (real_width > numchan){
                real_width = numchan;
            }
            convolve(singal_array, numchan, boxcar_kernel, real_width, signal_array_boxcar);
            real_center = argmax(signal_array_boxcar, numchan);
            if (abs((int)(real_center - single_bin_center)) < single_bin_width + real_width) {
                right_range += 1;
            }
            else{
		right_range += 1;
		flag += 1;
            }
	}else{
	    right_range += 1;
            flag += 1;
        }  
    }
    return right_range;
}


int find_left_range(float** matrix_normalized_work, int work_length, int numchan, int BASSET_minL, int min_BASSET_Ls){
    int left_range = 0, ii, jj, single_bin_width, real_width, single_bin_center, real_center, flag=0;
    float single_bin_array[numchan], single_bin_array_boxcar[numchan], singal_array[numchan], signal_array_boxcar[numchan];
    float boxcar_kernel[numchan];
    for (jj = 0; jj < numchan; jj++) {
        boxcar_kernel[jj] = 1.0;
    }
    float autocorrelation_result[2 * numchan - 1];
    
    while(left_range < work_length){
        if (flag > 2){
            break;
        }
        for (ii = 0; ii < numchan; ii++) {
           single_bin_array[ii] = matrix_normalized_work[work_length - left_range][ii];
        }
	    convolve(single_bin_array, numchan, boxcar_kernel, BASSET_minL, single_bin_array_boxcar);
        autocorrelation(single_bin_array_boxcar, numchan, autocorrelation_result);
        single_bin_width = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
        if (single_bin_width > min_BASSET_Ls / 2){
            if (single_bin_width > numchan){
                single_bin_width = numchan;
            }
            convolve(single_bin_array, numchan, boxcar_kernel, single_bin_width, single_bin_array_boxcar);
            single_bin_center = argmax(single_bin_array_boxcar, numchan);
            calculate_matrix_sum(matrix_normalized_work, 2 * work_length, numchan, 0, 
                        work_length - left_range, work_length + 1, singal_array);
            convolve(singal_array, numchan, boxcar_kernel, BASSET_minL, signal_array_boxcar);
            autocorrelation(signal_array_boxcar, numchan, autocorrelation_result);
            real_width = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
            if (real_width > numchan){
                real_width = numchan;
            }
            convolve(singal_array, numchan, boxcar_kernel, real_width, signal_array_boxcar);
            real_center = argmax(signal_array_boxcar, numchan);
            if (abs((int)(real_center - single_bin_center)) < single_bin_width + real_width) {
                left_range += 1;
            } 
            else{
		        left_range += 1;
                flag += 1;
            }
	}else{
	    left_range += 1;
            flag += 1; 
        } 
    }
    return left_range;
}


void find_frequency_range(float **data_pulse, int data_pulse_length, float **data_bg, 
                int data_bg_length, int numchan, int frequency_low, 
                int frequency_high, int *result_low, int *result_high) {
    // find high frequency
    float *SNRs = (float *)malloc((numchan - frequency_high) * sizeof(float));
    int frequency_high_tmp;
    float stds[numchan];

    get_stds(data_bg, data_bg_length, numchan, stds);
    for (int ii = 0; ii < numchan - frequency_high; ++ii) {
        float *line = (float *)malloc(data_pulse_length * sizeof(float));
        float *line_bg = (float *)malloc(data_bg_length * sizeof(float));

        // get line and line_bg
        calculate_matrix_sum(data_pulse, data_pulse_length, numchan, 1, frequency_low, frequency_high + ii, line);
        calculate_matrix_sum(data_bg, data_bg_length, numchan, 1, frequency_low, frequency_high + ii, line_bg);

        float SNR = (calculate_max(line, data_pulse_length) - calculate_mean(line_bg, data_bg_length))
                     / sum_of_squares_sqrt(stds, frequency_low, frequency_high + ii);
        SNRs[ii] = SNR;

        free(line);
        free(line_bg);
    }

    frequency_high_tmp = argmax(SNRs, numchan - frequency_high) + frequency_high;
    free(SNRs); 
   
    // find low frequency
    SNRs = (float *)malloc(frequency_low * sizeof(float));

    for (int ii = 0; ii < frequency_low; ++ii) {
        float *line = (float *)malloc(data_pulse_length * sizeof(float));
        float *line_bg = (float *)malloc(data_bg_length * sizeof(float));

        // get line and line_bg
        calculate_matrix_sum(data_pulse, data_pulse_length, numchan, 1, frequency_low - ii, frequency_high, line);
        calculate_matrix_sum(data_bg, data_bg_length, numchan, 1, frequency_low - ii, frequency_high, line_bg);

        float SNR = (calculate_max(line, data_pulse_length) - calculate_mean(line_bg, data_bg_length))
                     / sum_of_squares_sqrt(stds, frequency_low - ii, frequency_high);
        SNRs[ii] = SNR;

        free(line);
        free(line_bg);
    }

    *result_low = frequency_low - argmax(SNRs, frequency_low);
    *result_high = frequency_high_tmp;

    free(SNRs);
}


float sum_of_squares_sqrt(float arr[], int a, int b) {
    float sum = 0.0;
    for(int i = a; i < b; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrt(sum);
}


void get_stds(float** data_bg, int rows, int cols, float* stds) {
    float temp[rows]; // Temporary array to store column elements
    for(int jj = 0; jj < cols; jj++) {
        for(int kk = 0; kk < rows; kk++) {
            temp[kk] = data_bg[kk][jj]; // Copy column elements to temp
        }
        stds[jj] = calculate_standard_deviation(temp, rows); // Calculate standard deviation of the column
    }
}

void multi_trigger(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, 
                   int *BASSET_Ls, int BASSET_Ls_num, _Bool *trigger_flag) {
    int ii;
    int corr_Ls[BASSET_Ls_num + 1];
    int corr_Ls_num = BASSET_Ls_num + 1;
    corr_Ls[0] = calculate_min(BASSET_Ls, BASSET_Ls_num) / 2;
    for (ii = 1; ii < BASSET_Ls_num + 1; ii++){
        corr_Ls[ii] = BASSET_Ls[ii-1];
    }

    // Allocate memory for 2D arrays corr and corr_bg
    float **corr = (float **)malloc(corr_Ls_num * sizeof(float *));
    for (int i = 0; i < corr_Ls_num; ++i) {
        corr[i] = (float *)malloc(numchan * sizeof(float));
    }

    float **corr_bg = (float **)malloc(corr_Ls_num * sizeof(float *));
    for (int i = 0; i < corr_Ls_num; ++i) {
        corr_bg[i] = (float *)malloc(numchan * sizeof(float));
    }

    // Allocate memory for other 1D arrays based on numchan
    float bg[numchan];
    float matrix[numchan];

    // Main loop for processing
    for (ii = start_index; ii < numpts / one_ms_len - 2 * work_length; ii++) {
        dsamp_in_time(&data[numchan * ii * one_ms_len], numchan, bg_length * one_ms_len, matrix);
        dsamp_in_time(&data[numchan * (ii - 1 - (work_length + bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, bg);
        boxcar_correlation(matrix, numchan, corr_Ls, corr_Ls_num, corr);
        boxcar_correlation(bg, numchan, corr_Ls, corr_Ls_num, corr_bg);
        trigger_flag[ii] = trigger(corr, corr_bg, numchan, corr_Ls, corr_Ls_num);
    }

    // Free allocated memory for corr and corr_bg
    for (int i = 0; i < BASSET_Ls_num; ++i) {
        free(corr[i]);
        free(corr_bg[i]);
    }
    free(corr);
    free(corr_bg);
}

void multi_trigger_else(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, _Bool *trigger_flag, int BASSET_minL, int min_BASSET_Ls){
    int ii, jj;
    float autocorrelation_result[2 * numchan - 1], matrix[numchan], matrix_boxcar[numchan];
    float boxcar_kernel[numchan];
    int real_bandwidth;
    for (jj = 0; jj < numchan; jj++) {
        boxcar_kernel[jj] = 1.0;
    }
    for (ii = start_index; ii < numpts / one_ms_len - 2 * work_length; ii++){
        if (trigger_flag[ii]){
            dsamp_in_time(&data[numchan * ii * one_ms_len], numchan, bg_length * one_ms_len, matrix);
            convolve(matrix, numchan, boxcar_kernel, BASSET_minL, matrix_boxcar);
            autocorrelation(matrix_boxcar, numchan, autocorrelation_result);
            real_bandwidth = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;

            /* The ratio of bandwidth to ACF width is 1.13 for 
            ideal box-car shape pulses, and conservatively given as 1.5.*/
            if (numchan <= real_bandwidth || real_bandwidth <= min_BASSET_Ls / 1.5)
            {
                trigger_flag[ii] = 0;
            }
            
        }
    }
    
}

void multi_range_search(int start_index, int numpts, int one_ms_len, int work_length, 
                   float *data, int numchan, int bg_length, _Bool *trigger_flag, int BASSET_minL, int min_BASSET_Ls){
    int right_range, left_range, real_bandwidth;
    float tmp_len[bg_length], ds_len[numchan], autocorrelation_result[2 * numchan - 1];
    float singal_array[numchan], signal_array_boxcar[numchan], channel_mean[numchan];
    long long max_bin, ii, jj, mm;
    float **matrix_normalized_work;
    matrix_normalized_work = (float **)malloc((2 * work_length) * sizeof(float *));
    for (int i = 0; i < 2 * work_length; ++i) {
    	matrix_normalized_work[i] = (float *)malloc(numchan * sizeof(float));
    }
    float boxcar_kernel[numchan];
    for (jj = 0; jj < numchan; jj++) {
        boxcar_kernel[jj] = 1.0;
    }

    for (ii = start_index; ii < numpts / one_ms_len - 2 * work_length; ii++){
        if (trigger_flag[ii]){
            for(jj = 0; jj < bg_length; jj++){
                dsamp_in_time(&data[numchan * (ii + jj) * one_ms_len], numchan, one_ms_len, ds_len);
                tmp_len[jj] = calculate_sum(ds_len, numchan);
            }
            max_bin = argmax(tmp_len, bg_length) + ii;

            /* get channel_mean */
            dsamp_in_time(&data[numchan * (max_bin - (work_length + bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, ds_len);
            for(jj = 0; jj < numchan; jj++){
                channel_mean[jj] = ds_len[jj] / bg_length;
            }

            /* get matrix_normalized_work */
            for(jj = 0; jj < 2 * work_length; jj++){
                dsamp_in_time(&data[numchan * (max_bin - work_length + jj) * one_ms_len], numchan, one_ms_len, ds_len);
                for (mm = 0; mm < numchan; mm++){
                    matrix_normalized_work[jj][mm] = ds_len[mm] - channel_mean[mm];
                }
            }

            /* find right and left range */
            right_range = find_right_range(matrix_normalized_work, work_length, numchan, BASSET_minL, min_BASSET_Ls);
            left_range = find_left_range(matrix_normalized_work, work_length, numchan,  BASSET_minL, min_BASSET_Ls);

            /* get real_bandwidth */
            calculate_matrix_sum(matrix_normalized_work, 2 * work_length, numchan, 0, work_length - left_range, work_length + right_range, singal_array);
            convolve(singal_array, numchan, boxcar_kernel, BASSET_minL, signal_array_boxcar);
            autocorrelation(signal_array_boxcar, numchan, autocorrelation_result);
            real_bandwidth = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
            /* The ratio of bandwidth to ACF width is 1.13 for 
            ideal box-car shape pulses, and conservatively given as 1.5.*/
            if (numchan <= real_bandwidth || real_bandwidth <= min_BASSET_Ls / 1.5)
            {
                trigger_flag[ii] = 0;
            }
        }
    }
}

void adaptive_filter(float *data, int numpts, int numchan, int *BASSET_Ls, int BASSET_Ls_num, int BASSET_minL, int one_ms_len,
 float* outdata_BASSET, int totwrote, int downsamp, float dt, FILE* filter_inf_file){
    long long ii, jj, kk, mm, nn, start_index, max_bin;
    long long noise_bin = -1, noise_bin_before = -1, last_trigger_bound = -1;
    int work_length = 10, bg_length = 5, waiting_length = 30;
    int noise_bin_now = -work_length;
    int right_range, left_range, real_bandwidth;
    int data_pulse_length, data_bg_length, frequency_low, frequency_high;
    float noise_testing_len[waiting_length];
    float ds_len[numchan], bg[numchan], matrix[numchan], matrix_boxcar[numchan], channel_mean[numchan];
    float singal_array[numchan], signal_array_boxcar[numchan], stds[numchan];
    int corr_Ls_num = BASSET_Ls_num + 1;
    int corr_Ls[corr_Ls_num];
    corr_Ls[0] = calculate_min(BASSET_Ls, BASSET_Ls_num) / 2;
    for (ii = 1; ii < BASSET_Ls_num + 1; ii++){
        corr_Ls[ii] = BASSET_Ls[ii-1];
    }
    float **corr;
    float **corr_bg;
    corr = (float **)malloc(corr_Ls_num * sizeof(float *));
    for (int i = 0; i < corr_Ls_num; ++i) {
    	corr[i] = (float *)malloc(numchan * sizeof(float));
    }
    corr_bg = (float **)malloc(corr_Ls_num * sizeof(float *));
    for (int i = 0; i < corr_Ls_num; ++i) {
    	corr_bg[i] = (float *)malloc(numchan * sizeof(float));
    }
    float autocorrelation_result[2 * numchan - 1];
    float boxcar_kernel[numchan];
    for (jj = 0; jj < numchan; jj++) {
        boxcar_kernel[jj] = 1.0;
    }
    float all_band_series_mean, signal_band_series_mean, tmp, STD_ratio;
    float tmp_len[bg_length];
    float **matrix_normalized_work;
    float **data_pulse;
    float **data_bg;
    matrix_normalized_work = (float **)malloc((2 * work_length) * sizeof(float *));
    for (int i = 0; i < 2 * work_length; ++i) {
    	matrix_normalized_work[i] = (float *)malloc(numchan * sizeof(float));
    }
    data_pulse = (float **)malloc((2 * work_length) * sizeof(float *));
    for (int i = 0; i < 2 * work_length; ++i) {
        data_pulse[i] = (float *)malloc(numchan * sizeof(float));
    }
    data_bg = (float **)malloc((work_length + bg_length) * sizeof(float *));
    for (int i = 0; i < work_length + bg_length; ++i) {
        data_bg[i] = (float *)malloc(numchan * sizeof(float));
    }
    float all_band_series[work_length + bg_length];
    float signal_band_series[work_length + bg_length];
    char filter_inf[200];
    float time;
    //char name[40];
    _Bool trigger_flag[numpts / one_ms_len];
    _Bool trigger_twice;
    _Bool no_need_trigger;
    int min_BASSET_Ls = calculate_min(BASSET_Ls, BASSET_Ls_num);

    /* initialization */   
    start_index = waiting_length;
    while (1){
        for (ii = (start_index - waiting_length); ii < start_index; ii++){
            dsamp_in_time(&data[numchan * ii * one_ms_len], numchan, one_ms_len, ds_len);
            noise_testing_len[ii - (start_index - waiting_length)] = calculate_sum(ds_len, numchan);
        }
        if (get_SNR(noise_testing_len, waiting_length, noise_testing_len, waiting_length) > 3){
            start_index += 1;
        }
        else{
            break;
        }
    }

    for (ii = 0; ii < start_index; ii++){
        for (kk = 0; kk < one_ms_len; kk++){
            outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk)], numchan);
        }
    }
    
    /*trigger first*/
    multi_trigger(start_index, numpts, one_ms_len, work_length, data, numchan, bg_length, BASSET_Ls, BASSET_Ls_num, trigger_flag);
    multi_trigger_else(start_index, numpts, one_ms_len, work_length, data, numchan, bg_length, trigger_flag, BASSET_minL, min_BASSET_Ls);
    multi_range_search(start_index, numpts, one_ms_len, work_length, data, numchan, bg_length, trigger_flag, BASSET_minL, min_BASSET_Ls);

    /* start loop */
    for (ii = start_index; ii < numpts / one_ms_len - 2 * work_length; ii++){
        
        /* trigger */
        if (!trigger_flag[ii]) {
            for (kk = 0; kk < one_ms_len; kk++){
                outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk)], numchan);
            }
        }

        /* maybe there is a pulse */
        else{

            /* trigger twice*/
            dsamp_in_time(&data[numchan * ii * one_ms_len], numchan, bg_length * one_ms_len, matrix);
            if (ii - 1 - waiting_length < noise_bin_now) {
                dsamp_in_time(&data[numchan * (noise_bin - (work_length + 2 * bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, bg);
                no_need_trigger = 0;
            }
            else{
                dsamp_in_time(&data[numchan * (ii - 1 - (work_length + bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, bg);
                no_need_trigger = 1;
                trigger_twice = 1;
            }
            boxcar_correlation(matrix, numchan, corr_Ls, corr_Ls_num, corr);
            boxcar_correlation(bg, numchan, corr_Ls, corr_Ls_num, corr_bg);
            if (!no_need_trigger)
            {
                trigger_twice = trigger(corr, corr_bg, numchan, corr_Ls, corr_Ls_num);
            }
            
            if (!trigger_twice){
                for (kk = 0; kk < one_ms_len; kk++){
                    outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk)], numchan);
                }
            }
            
            /* using the autocorrection */
            else{
                if (!no_need_trigger){
                    convolve(matrix, numchan, boxcar_kernel, BASSET_minL, matrix_boxcar);
                    autocorrelation(matrix_boxcar, numchan, autocorrelation_result);
                    real_bandwidth = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;
                }else{
                    real_bandwidth = -1;
                }                

                /* find the max bin and get range */
                /* The ratio of bandwidth to ACF width is 1.13 for 
                ideal box-car shape pulses, and conservatively given as 1.5.*/
                if ((numchan > real_bandwidth && real_bandwidth > min_BASSET_Ls / 1.5) || no_need_trigger) {
                    for(jj = 0; jj < bg_length; jj++){
                        dsamp_in_time(&data[numchan * (ii + jj) * one_ms_len], numchan, one_ms_len, ds_len);
                        tmp_len[jj] = calculate_sum(ds_len, numchan);
                    }
                    max_bin = argmax(tmp_len, bg_length) + ii;

                    /* get channel_mean */
                    if (max_bin - work_length - waiting_length < noise_bin_now){
                        dsamp_in_time(&data[numchan * (noise_bin - (work_length + bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, ds_len);
                        for(jj = 0; jj < numchan; jj++){
                            channel_mean[jj] = ds_len[jj] / bg_length;
                        }
                    }
                    else{
                        dsamp_in_time(&data[numchan * (max_bin - (work_length + bg_length)) * one_ms_len], numchan, bg_length * one_ms_len, ds_len);
                        for(jj = 0; jj < numchan; jj++){
                            channel_mean[jj] = ds_len[jj] / bg_length;
                        }
                    }

                    /* get matrix_normalized_work */
                    for(jj = 0; jj < 2 * work_length; jj++){
                        dsamp_in_time(&data[numchan * (max_bin - work_length + jj) * one_ms_len], numchan, one_ms_len, ds_len);
                        for (mm = 0; mm < numchan; mm++){
                            matrix_normalized_work[jj][mm] = ds_len[mm] - channel_mean[mm];
                        }
                    }

                    /* find right and left range */
                    right_range = find_right_range(matrix_normalized_work, work_length, numchan, BASSET_minL, min_BASSET_Ls);
                    left_range = find_left_range(matrix_normalized_work, work_length, numchan, BASSET_minL, min_BASSET_Ls);
                    if(max_bin - left_range < last_trigger_bound){
                        left_range = max_bin - last_trigger_bound;    
                    }

                    /* get real_bandwidth */
                    calculate_matrix_sum(matrix_normalized_work, 2 * work_length, numchan, 0, work_length - left_range, work_length + right_range, singal_array);
                    convolve(singal_array, numchan, boxcar_kernel, BASSET_minL, signal_array_boxcar);
                    autocorrelation(signal_array_boxcar, numchan, autocorrelation_result);
                    real_bandwidth = (int) fit_gaussian(autocorrelation_result, 2 * numchan - 1) * 2;

                    /* Determine if it is RFI */
                    /* The ratio of bandwidth to ACF width is 1.13 for 
                    ideal box-car shape pulses, and conservatively given as 1.5.*/
                    if (numchan > real_bandwidth && real_bandwidth > min_BASSET_Ls / 1.5) {

                    /* get initial frequency_low and frequency_high */
                        convolve(singal_array, numchan, boxcar_kernel, real_bandwidth, signal_array_boxcar);
                        frequency_low = argmax(signal_array_boxcar, numchan) - real_bandwidth / 2;
                        frequency_low = (frequency_low > 1) ? frequency_low : 1;
                        frequency_high = frequency_low + real_bandwidth;
                        frequency_high = (frequency_high < numchan - 1) ? frequency_high : numchan - 1;

                        /* get data_pulse and data_bg */
                        if(max_bin - left_range - waiting_length < noise_bin_now){
                            data_pulse_length = left_range + right_range;
                            for (mm = 0; mm < data_pulse_length; ++mm) {
                                dsamp_in_time(&data[numchan * (max_bin - left_range + mm) * one_ms_len], numchan, one_ms_len, ds_len);
                                for (nn = 0; nn < numchan; ++nn) {
                                    data_pulse[mm][nn] = ds_len[nn];
                                }
                            }
                            data_bg_length = work_length + bg_length;
                            for (mm = 0; mm < data_bg_length; ++mm) {
                                dsamp_in_time(&data[numchan * (noise_bin - (work_length + 3 * bg_length) + mm) * one_ms_len], numchan, one_ms_len, ds_len);
                                for (nn = 0; nn < numchan; ++nn) {
                                    data_bg[mm][nn] = ds_len[nn];
                                }
                            }
                        }
                        else{
                            data_pulse_length = left_range + right_range;
                            for (mm = 0; mm < data_pulse_length; ++mm) {
                                dsamp_in_time(&data[numchan * (max_bin - left_range + mm) * one_ms_len], numchan, one_ms_len, ds_len);
                                for (nn = 0; nn < numchan; ++nn) {
                                    data_pulse[mm][nn] = ds_len[nn];
                                }
                            }
                            data_bg_length = work_length + bg_length;
                            for (mm = 0; mm < data_bg_length; ++mm) {
                                dsamp_in_time(&data[numchan * (max_bin - (work_length + 3 * bg_length) + mm) * one_ms_len], numchan, one_ms_len, ds_len);
                                for (nn = 0; nn < numchan; ++nn) {
                                    data_bg[mm][nn] = ds_len[nn];
                                }
                            }
                        }
                        
                        /* find frequency range of the pulse */
                        find_frequency_range(data_pulse, data_pulse_length, data_bg, data_bg_length, 
                        numchan, frequency_low, frequency_high, &frequency_low, &frequency_high);                    
                        
                        /* normalization */
                        calculate_matrix_sum(data_bg, data_bg_length, numchan, 1, 0, numchan, all_band_series);
                        calculate_matrix_sum(data_bg, data_bg_length, numchan, 1, frequency_low, frequency_high, signal_band_series);
                        get_stds(data_bg, data_bg_length, numchan, stds);
                        STD_ratio = sum_of_squares_sqrt(stds, 0, numchan) / sum_of_squares_sqrt(stds, frequency_low, frequency_high);
                        all_band_series_mean = calculate_mean(all_band_series, data_bg_length) / one_ms_len;
                        signal_band_series_mean = calculate_mean(signal_band_series, data_bg_length) / one_ms_len;

                        /* get the time series */ 
                        for (jj = ii; jj < max_bin - left_range; jj++){
                            for (kk = 0; kk < one_ms_len; kk++){
                                outdata_BASSET[jj * one_ms_len + kk] = calculate_sum(&data[numchan * (jj * one_ms_len + kk) ], numchan);
                            }
                        }

                        for (jj = 0; jj < data_pulse_length; jj++){
                            for (kk = 0; kk < one_ms_len; kk++){
                                tmp = calculate_sum(&data[numchan * ((max_bin - left_range + jj) * one_ms_len + kk) + frequency_low ], frequency_high - frequency_low);
                                tmp = (tmp - signal_band_series_mean) * STD_ratio + all_band_series_mean;
                                outdata_BASSET[(max_bin - left_range + jj) * one_ms_len + kk] = tmp;
                            }
                        }

                        /* write filter inf */
                        time = (totwrote + max_bin * one_ms_len) * downsamp * dt;
                        sprintf(filter_inf, "time:%.3f max_bin:%lld range:%lld-%lld  frquency:%d-%d real_bandwidth:%d STD_ratio:%.3f\n", time,
                        max_bin, max_bin - left_range, max_bin + right_range, frequency_low, frequency_high, real_bandwidth, STD_ratio);
                        fputs(filter_inf, filter_inf_file);

                        /* Handling the trigger_flag */
                        for (jj = 0; jj < waiting_length + 1; jj++){
                            trigger_flag[max_bin + right_range + jj] = 1;    
                        }

                        /* Handling the noise bin */
                        if (noise_bin_now == -work_length && noise_bin_before == -1) {
                            noise_bin_before = max_bin + right_range;
                            noise_bin_now = max_bin + right_range;
                            noise_bin = max_bin - left_range;
                        } else {
                            noise_bin_now = max_bin + right_range;
                        }
                        
                        if (max_bin - left_range - noise_bin_before > waiting_length){
                            noise_bin = max_bin - left_range;
                            noise_bin_before = max_bin + right_range;
                        } else {
                            noise_bin_before = max_bin + right_range;
                        }
                        last_trigger_bound = noise_bin_now;
                        ii = max_bin + right_range;
                    }else{
                        for (kk = 0; kk < one_ms_len; kk++){
                            outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk)], numchan);
                        }
                    }
                }else{
                    for (kk = 0; kk < one_ms_len; kk++){
                        outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk)], numchan);
                    }
                }
            }
        }
    }

    /* Handle the end */
    for (ii = numpts / one_ms_len - 2 * work_length; ii < numpts / one_ms_len; ii++){
        for (kk = 0; kk < one_ms_len; kk++){
            outdata_BASSET[ii * one_ms_len + kk] = calculate_sum(&data[numchan * (ii * one_ms_len + kk) ], numchan);
        }
    }
    /* free */
    for (int i = 0; i < BASSET_Ls_num; ++i) {
        free(corr[i]);
    }
    free(corr);
    for (int i = 0; i < BASSET_Ls_num; ++i) {
    	free(corr_bg[i]);
    }
    free(corr_bg);
    for (int i = 0; i < 2 * work_length; ++i) {
    	free(matrix_normalized_work[i]);
    }
    free(matrix_normalized_work);
    for (int i = 0; i < 2 * work_length; ++i) {
    	free(data_pulse[i]);
    }
    free(data_pulse);
    for (int i = 0; i < work_length + bg_length; ++i) {
    	free(data_bg[i]);
    }
    free(data_bg);
}

//debug function
void save_array_to_txt(float** array, int m, int n, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%lf ", array[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}


void string_to_array(const char *str, int *array, int *len) {
    *len = 1;
    for (const char *p = str; *p; ++p) {
        if (*p == ',') {
            ++(*len);
        }
    } 
    const char *p = str;
    
    for (int i = 0; i < *len; ++i) {
        char *endptr;
        int val = strtol(p, &endptr, 10);
        if (endptr == p) {
            fprintf(stderr, "Error: invalid input string\n");
            exit(EXIT_FAILURE);
        }
        array[i] = val;
        p = endptr;
        if (*p == ',') {
            ++p;
        }
    }
}
